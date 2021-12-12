1 kib = 1024 = 2^10
1mib = 1024 * 1024 = 2^(10 + 10) = 2^20
64mib =1024 * 1024 * 64 =  2^(10 + 10 + 6) = 2^26
1gib = 1024 * 1024 * 1024 = 2^(10 + 10 + 10) = 2^30

If we have 64 mib arenas
then the number of arenas is:

2^48 / 2^26
= 2^(48 - 26)
= 2^24
if we put this in a bitmap, then the size of the bitmap is
2^24 / 8
= 2^24 / 2^3
= 2^(24 - 3)
= 2^21 bytes
= 2^(20 + 1) bytes
= 2^20 * 2^1
= 1mebibyte * 2
= 2 mib

2 1
4 2
8 3
16 4
