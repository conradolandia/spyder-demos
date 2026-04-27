#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 19:02:38 2026

@author: andi
"""

def gcd(a, b):
    # a > b
    if b == 0:
        return a
    else:
        return gcd(b, a % b)


def cubic(c, t0):
    t = t0
    while abs(t**3 - c) > 1e-10:
        t = (2 * t + c / (t * t)) / 3
    return t


def millerRabin(n):
    n_1 = n - 1
    t = n_1
    s = 0

    boo = False

    # decomposition n-1 = 2^s * t
    while t % 2 == 0:
        t = t // 2
        s = s + 1

    for b in range(2, n):
        if gcd(b, n) == 1:
            y = (b ** t) % n

            if y == 1:
                boo = True
            else:
                for i in range(0, s):
                    if y == n - 1:
                        boo = True
                    else:
                        y = (y ** 2) % n

    return boo


# entry point
if __name__ == "__main__":
    print("Miller-Rabin(15):", millerRabin(15))
    print("Cubic root approx:", cubic(2107, 10))
