"""Galois Field GF(2^8) arithmetic and Reed-Solomon (N, K) Forward Error Correction."""

from dataclasses import dataclass
import numpy as np


class GaloisField256:
    """Galois Field GF(2^8) arithmetic with precomputed tables."""

    def __init__(self, prim_poly: int = 0x11D, generator: int = 2) -> None:
        self.prim_poly = prim_poly
        self.exp = [0] * 512
        self.log = [0] * 256

        x = 1
        for i in range(255):
            self.exp[i] = x
            self.log[x] = i
            x <<= 1
            if x & 0x100:
                x ^= self.prim_poly

        for i in range(255, 512):
            self.exp[i] = self.exp[i - 255]

    def add(self, a: int, b: int) -> int:
        return a ^ b

    def sub(self, a: int, b: int) -> int:
        return a ^ b

    def mul(self, a: int, b: int) -> int:
        if a == 0 or b == 0:
            return 0
        return self.exp[self.log[a] + self.log[b]]

    def div(self, a: int, b: int) -> int:
        if b == 0:
            raise ZeroDivisionError("GF(2^8) division by zero")
        if a == 0:
            return 0
        return self.exp[(self.log[a] - self.log[b] + 255) % 255]

    def inv(self, a: int) -> int:
        if a == 0:
            raise ZeroDivisionError("GF(2^8) inverse of zero")
        return self.exp[255 - self.log[a]]

    def poly_mul(self, p: list[int], q: list[int]) -> list[int]:
        """Multiply two polynomials over GF(2^8)."""
        r = [0] * (len(p) + len(q) - 1)
        for i, a in enumerate(p):
            for j, b in enumerate(q):
                r[i + j] ^= self.mul(a, b)
        return r

    def poly_eval(self, poly: list[int], x: int) -> int:
        """Evaluate polynomial at point x using Horner's method."""
        y = poly[0]
        for coeff in poly[1:]:
            y = self.add(self.mul(y, x), coeff)
        return y


@dataclass
class ReedSolomonResult:
    """Result from Reed-Solomon decoding."""

    success: bool
    corrected_data: bytes
    error_count: int
    error_positions: list[int]
    notes: str = ""


class ReedSolomonCodec:
    """Reed-Solomon (N, K) Codec over GF(2^8).

    N: Codeword length (e.g. 255)
    K: Message length (e.g. 223 or 239)
    2t = N - K parity symbols (can correct up to t symbol errors).
    """

    def __init__(self, n: int = 255, k: int = 223, fcr: int = 0) -> None:
        if n > 255 or k >= n or k <= 0:
            raise ValueError(f"Invalid RS({n}, {k}) parameters")
        self.n = n
        self.k = k
        self.nsym = n - k
        self.fcr = fcr
        self.gf = GaloisField256()

        # Compute generator polynomial g(x) = prod_{i=0}^{nsym-1} (x - alpha^(i + fcr))
        self.gen = [1]
        for i in range(self.nsym):
            root = self.gf.exp[i + self.fcr]
            self.gen = self.gf.poly_mul(self.gen, [1, root])

    def encode(self, msg: bytes | np.ndarray) -> bytes:
        """Systematic Reed-Solomon encoding."""
        msg_bytes = bytes(msg)
        if len(msg_bytes) > self.k:
            raise ValueError(f"Message length {len(msg_bytes)} exceeds capacity {self.k}")

        # Pad message if shorter than K
        pad_len = self.k - len(msg_bytes)
        padded_msg = (b"\x00" * pad_len) + msg_bytes

        out = list(padded_msg) + [0] * self.nsym
        for i in range(len(padded_msg)):
            feedback = out[i]
            if feedback != 0:
                for j in range(1, len(self.gen)):
                    out[i + j] ^= self.gf.mul(self.gen[j], feedback)

        # Result is padded message + parity
        codeword = padded_msg + bytes(out[len(padded_msg):])
        # Return only the relevant slice
        return codeword[pad_len:]

    def decode(self, codeword: bytes | np.ndarray) -> ReedSolomonResult:
        """Decode Reed-Solomon codeword using Berlekamp-Massey & Forney algorithms."""
        c = list(bytes(codeword))
        if len(c) > self.n:
            raise ValueError(f"Codeword length {len(c)} exceeds N={self.n}")

        pad_len = self.n - len(c)
        full_c = ([0] * pad_len) + c

        # 1. Compute syndromes S_i = C(alpha^(i + fcr))
        syndromes = [0] * self.nsym
        has_error = False
        for i in range(self.nsym):
            val = self.gf.poly_eval(full_c, self.gf.exp[i + self.fcr])
            syndromes[i] = val
            if val != 0:
                has_error = True

        if not has_error:
            # Clean codeword, extract message part
            msg = bytes(full_c[pad_len : self.k])
            return ReedSolomonResult(
                success=True,
                corrected_data=msg,
                error_count=0,
                error_positions=[],
                notes="Zero syndromes: No errors detected.",
            )

        # 2. Berlekamp-Massey algorithm for error locator polynomial Lambda(x)
        # C(x) is current polynomial, B(x) is previous discrepancy polynomial
        C_poly = [1]
        B_poly = [1]
        L = 0
        m = 1
        b_val = 1

        for n_idx in range(self.nsym):
            # Compute discrepancy d
            d = syndromes[n_idx]
            for i in range(1, len(C_poly)):
                d ^= self.gf.mul(C_poly[i], syndromes[n_idx - i])

            if d == 0:
                m += 1
            else:
                scale = self.gf.div(d, b_val)
                # T(x) = C(x) - scale * x^m * B(x)
                shift_B = ([0] * m) + [self.gf.mul(x, scale) for x in B_poly]
                # Pad to same length for addition
                max_len = max(len(C_poly), len(shift_B))
                C_padded = C_poly + [0] * (max_len - len(C_poly))
                B_padded = shift_B + [0] * (max_len - len(shift_B))
                T_poly = [self.gf.add(C_padded[i], B_padded[i]) for i in range(max_len)]

                if 2 * L <= n_idx:
                    L = n_idx + 1 - L
                    B_poly = C_poly
                    b_val = d
                    m = 1
                else:
                    m += 1
                C_poly = T_poly

        # Number of errors indicated by deg(Lambda)
        # Trim trailing zeros
        while len(C_poly) > 1 and C_poly[-1] == 0:
            C_poly.pop()

        error_count = len(C_poly) - 1
        if error_count * 2 > self.nsym:
            return ReedSolomonResult(
                success=False,
                corrected_data=bytes(c[: len(c) - self.nsym]),
                error_count=error_count,
                error_positions=[],
                notes=f"Uncorrectable: error count {error_count} exceeds capacity {self.nsym // 2}.",
            )

        # 3. Chien Search: Find roots of Lambda(x)
        error_positions: list[int] = []
        for i in range(self.n):
            # Evaluate Lambda at alpha^(-i)
            inv_alpha = self.gf.exp[(255 - i) % 255]
            val = self.gf.poly_eval(C_poly[::-1], inv_alpha)
            if val == 0:
                pos = self.n - 1 - i
                error_positions.append(pos)

        if len(error_positions) != error_count:
            return ReedSolomonResult(
                success=False,
                corrected_data=bytes(c[: len(c) - self.nsym]),
                error_count=error_count,
                error_positions=[],
                notes="Uncorrectable: Chien search root count does not match locator polynomial degree.",
            )

        # 4. Forney Algorithm: Compute error values
        # Omega(x) = (Syndromes(x) * Lambda(x)) mod x^nsym
        omega = self.gf.poly_mul(syndromes, C_poly)[: self.nsym]

        # Formal derivative of Lambda(x)
        lambda_deriv = [0] * len(C_poly)
        for i in range(1, len(C_poly)):
            if i % 2 == 1:  # odd powers only in GF(2)
                lambda_deriv[i - 1] = C_poly[i]
        while len(lambda_deriv) > 1 and lambda_deriv[-1] == 0:
            lambda_deriv.pop()

        # Apply corrections
        corrected_full = list(full_c)
        for pos in error_positions:
            x_inv = self.gf.exp[(255 - (self.n - 1 - pos)) % 255]
            # Omega(X^-1)
            num = self.gf.poly_eval(omega[::-1], x_inv)
            # Lambda'(X^-1)
            denom = self.gf.poly_eval(lambda_deriv[::-1], x_inv)
            if denom == 0:
                continue
            err_val = self.gf.div(num, denom)
            # Factor for fcr != 0: alpha^(-(fcr - 1))
            if self.fcr != 1:
                shift_factor = self.gf.exp[((1 - self.fcr) * (self.n - 1 - pos)) % 255]
                err_val = self.gf.mul(err_val, shift_factor)

            corrected_full[pos] ^= err_val

        # Verify syndromes of corrected codeword
        valid = True
        for i in range(self.nsym):
            if self.gf.poly_eval(corrected_full, self.gf.exp[i + self.fcr]) != 0:
                valid = False
                break

        actual_msg = bytes(corrected_full[pad_len : self.k])
        # Shift positions relative to input codeword
        rel_positions = [p - pad_len for p in error_positions if p >= pad_len]

        if valid:
            return ReedSolomonResult(
                success=True,
                corrected_data=actual_msg,
                error_count=len(rel_positions),
                error_positions=rel_positions,
                notes=f"Corrected {len(rel_positions)} symbol error(s).",
            )
        else:
            return ReedSolomonResult(
                success=False,
                corrected_data=bytes(c[: len(c) - self.nsym]),
                error_count=len(rel_positions),
                error_positions=rel_positions,
                notes="Correction verification failed.",
            )
