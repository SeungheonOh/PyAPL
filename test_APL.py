from APL import *

a = APLArray([1, 2, 3, 4, 5])
b = APLArray([2, 3, 4, 5, 6])
c = A([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
m = Rho([2, 3, 4, 5, 6], Iota(2 * 3 * 4 * 5 * 6))
t = Rho([2, 3, 4, 5, 6], Iota(6))
x = Rho([2, 2, 3], Iota(2*2*3))
u = Rho([2, 3, 4], Iota(10))
i = Rho([4, 3, 2], Iota(10))


def test_Basic():
    assert m[2, 1, 2, 1, 2] == 392
    assert m[2, 1, 2, 1] == APLArray([391,392,393,394,395,396], [6])
    assert Rho([2, 3, 3], Iota(18)) == APLArray([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], [2, 3, 3])

def test_Types():
    assert APLArray([1]).shape == [0]

    assert Plus([1], [2]) == APLArray([3])
    assert Plus([1], [2]).shape == [0]
    assert Reduce(Plus, [1, 2, 3]) == APLArray([6])

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

def test_Drop():
    assert Drop([1, -1, 1], x) == APLArray([8, 9], [1,1,2])
    assert Drop([1, 1, 3, 4], m) == APLArray([595,596,597,598,599,600,715,716,717,718,719,720], [1, 2, 1, 1, 6])
    assert Drop([-1, 1, -3, 2], m) == APLArray([133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,253,254,255,256,257,258,259,260,261,262,263,264,265,266,267,268,269,270],[1, 2, 1, 3, 6])
    assert Drop([1], [1, 2, 3]) == APLArray([2, 3])

def test_Reduce():
    assert type(Reduce(Plus, APLArray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))) == APLArray
    assert Reduce(Plus, [1, 2, 3]) == APLArray([6])
    assert Reduce(Plus, APLArray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])).shape == [0]
    assert Reduce(Plus, APLArray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])) == APLArray([55])
    assert Reduce(Max, APLArray([1, 6, 7, 8, 9, 10, 2, 3, 4, 5])) == APLArray([10])
    assert Reduce(Plus, x) == APLArray([6,15,24,33], [2, 2])
    assert Reduce(Divi, m)

def test_Dot():
    assert dot(Max, Mult)(u, i) == APLArray([36,40,18,20,21,24,72,80,54,60,49,56,70,80,90,100,45,54,54,60,36,40,35,40,90,100,72,80,63,72,36,40,18,20,21,24], [2, 3, 3, 2])
    assert dot(Plus, Mult)(Rho([2, 3], Iota(6)), Rho([3, 2], Iota(5))) == APLArray([22,13,49,34], [2,2])

def test_Average():
    avg = lambda a: Divi(Reduce(Plus, a), len(a))
    assert avg(c) == A([5.5])
    assert avg(b) == A([4])
