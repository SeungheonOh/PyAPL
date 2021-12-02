from APL import *

a = APLArray([1, 2, 3, 4, 5])
b = APLArray([2, 3, 4, 5, 6])
m = Rho([2, 3, 4, 5, 6], Iota(2 * 3 * 4 * 5 * 6))
t = Rho([2, 3, 4, 5, 6], Iota(6))
x = Rho([2, 2, 3], Iota(2*2*3))

def test_Basic():
    assert m[2, 1, 2, 1, 2] == 392
    assert m[2, 1, 2, 1] == APLArray([391,392,393,394,395,396], [6])
    assert Rho([2, 3, 3], Iota(18)) == APLArray([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], [2, 3, 3])

def test_Arith():
    assert Minu(m) == APLArray(fmap(lambda a: -1*a, m.arr), m.shape)
    assert Plus(a, b) == APLArray([3, 5, 7, 9, 11])
    assert Mult(a, b) == APLArray(zipWith(lambda a, b: a*b, a.arr, b.arr))

def test_fill():
    assert Rho([2, 3, 4], Iota(100000)) == Rho([2, 3, 4], Iota(2*3*4))
    assert Rho([2, 3, 4], Iota(2)) == Rho([2, 3, 4], [1, 2] * 24)

def test_Split():
    assert Split(x) == APLArray([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], [2, 2])
    assert Split(x, axis=1) == APLArray([[1,7],[2,8],[3,9],[4,10],[5,11],[6,12]], [2, 3])

