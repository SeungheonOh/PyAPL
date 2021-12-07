# -*- coding: utf-8 -*- 
# TODO APLize function names
# TODO Better printing
import math

id = lambda a: a
fmap = lambda f, a: list(map(f, a))
reduce = lambda f, a: 0 if len(a) == 0 else f(a[0], reduce(f, a[1:])) if len(a) > 1 else a[0]
zipWith = lambda f, a, b: [f(a[0], b[0])] + zipWith(f, a[1:], b[1:]) if len(a) > 1 and len(b) > 1 else [f(a[0], b[0])]
zip3With = lambda f, a, b, c: [f(a[0], b[0], c[0])] + zip3With(f, a[1:], b[1:], c[1:]) if len(a) > 1 and len(b) > 1 else [f(a[0], b[0], c[0])]
numstr = lambda f: str(f) if not isinstance(f, float) else "{:.2f}".format(f)
indexs = lambda arr, opt: fmap(lambda a: indexs(a, opt[1:]), fmap(lambda a: arr + [a], opt[0])) if len(opt) > 0 else arr
flatten = lambda a, d: flatten(reduce(lambda a, b: a+b, a), d-1) if d > 0 else a

# returns unicode string
def box(msg, title=None):
  lines = fmap(lambda a: a.rstrip(), msg.split("\n"))
  row = max(fmap(lambda a: len(a), lines))
  if title:
    title = title[:row]
    t = ''.join(['┌'] + [title] + ['─'*(row - len(title))] + ['┐'])
  else: 
    t = ''.join(['┌'] + ['─'*row] + ['┐'])
  b = ''.join(['└'] + ['─'*row] + ['┘'])
  middle = "\n" + "\n".join(fmap(lambda a: "│{:{}s}│".format(a, row), lines)) + "\n"
  return t + middle + b

class APLError(Exception):
  pass

class APLArray:
  ## NOTE
  # self.arr is ALWAYS a vector
  # self.shape will determine the dementions of data
  ## TODO 
  # refactor shape, remove prefixes of one

  # shapes orders from largest demention to the least demention
  def __init__(self, arr, shape=None, box=True):
    self.box = box
    if isinstance(arr, int): 
      self.arr = [arr]
      self.shape = [0]
      return
    else:
      self.arr = arr

    if shape == [] or shape == [0] or len(arr) == 1:
      self.shape = [0] # array with shape 0 should be considered as a list
    else:
      self.shape = [len(arr)] if not shape else shape
    
  def __eq__(self, a):
    if not isinstance(a, APLArray):
      return
    return self.shape == a.shape and self.arr == a.arr

  def __str__(self):
    # Redo this to support array in array
    longest = max(fmap(lambda a: numstr(a), self.arr), key=lambda d: len(d))
    fmt = lambda a: '{0: >{width}}'.format(numstr(a), width=len(longest))
    mkStr = lambda arr: " ".join(fmap(fmt, arr.arr)) if len(arr.shape) == 1 else box("\n".join(fmap(lambda a: mkStr(arr.at(a)), range(1, arr.shape[0]+1))), title=",".join(fmap(str, arr.shape)))
    return mkStr(self).rstrip() + "\n"
  
  def __repr__(self):
    return str(self)

  # Let's make it iterable
  def __iter__(self):
    if self.shape == [0]:
      return iter(self.arr)
    self.iterIndex = 0
    return self
  
  def __next__(self):
    if self.iterIndex >= self.shape[0]:
      raise StopIteration
    self.iterIndex += 1
    return self[self.iterIndex]

  def __getitem__(self, key):
    if isinstance(key, slice):
      raise APLError("USE DROP INSTEAD")
    elif isinstance(key, int):
      return self.at(key)
    return self.at(*list(key))
  
  def __len__(self):
    return self.shape[0]
  
  # Fill empty spaces to fufill shape
  def fill(self): 
    need = reduce(lambda a, b: a*b, self.shape)
    if need < len(self.arr): # good
      self.arr = self.arr[:need] # clean up
      return self
    self.arr = (self.arr * need)[:need] # Wacky but whatever
    return self
  
  # Why not
  def mapAll(self, f):
    return APLArray(fmap(f, self.arr), self.shape)
  
  # value if singleton, None if not
  def singleton(self):
    return self.arr[0] if self.shape == [0] else None
  
  def vectorize(self):
    return self.arr

  # indexs orders from largest demention to the least demention 
  def at(self, *indexs): 
    ## Validity
    if len(indexs) > len(self.shape):
      raise APLError("INDEX ERROR", indexs)
    # Negative indexs are invalid
    if not reduce(lambda a, b: a and b, fmap(lambda a: a > 0, indexs)):
      raise APLError("INDEX ERROR", indexs)
    # Indexs that are out of bound are invalid
    if not reduce(lambda a, b: a and b, zipWith(lambda a, b: a <= b, indexs, self.shape)):
      raise APLError("INDEX ERROR", indexs)

    # Make multiplier based on shape
    shapeMultiplier = fmap(lambda l: reduce(lambda a, b: a * b, self.shape[l:])
                          , range(1, len(self.shape))) + [1]

    ## Apply multiplier
    indx = sum(zipWith(lambda a, b: (a-1) * b, indexs, shapeMultiplier))

    ## Amount
    newshape = self.shape[len(indexs):] if self.shape[len(indexs):] != [] else [1]
    amount = reduce(lambda a, b: a * b, newshape)


    ## Brand New Array
    if newshape == [1]: # Prevents nesting
      return self.arr[indx]
    return APLArray(self.arr[indx:indx+amount], newshape)

A = APLArray

# Let's make life easier
def APLize(arr):
  if isinstance(arr, list):
    return APLArray(arr)
  if isinstance(arr, APLArray):
    return arr
  if isinstance(arr, int):
    return APLArray([arr])
  if isinstance(arr, float):
    return APLArray([arr])

def Rho(left, right=None):
  left = APLize(left)
  if right != None: # Dyadic
    right = APLize(right)
    if len(left.shape) == 1:
      return APLArray(right.arr, left.arr).fill()
    else:
      raise APLError("RANK ERROR", left)
  else: # Monadic
    return left.shape

def Split(apl, axis=-1):
  apl = APLize(apl)
  if len(apl.shape) == 1:
    return APLArray(apl.arr, [])

  if axis == -1:
    axis = len(apl.shape) - 1
  else:
    axis = axis - 1
  # axis that are not fixed
  newshape = apl.shape[:axis] + apl.shape[axis+1:]

  # Requirements
  ## list of possible indexs of all dementions except axis
  ## like [[1, 2], [1, 2, 3]]
  req = fmap(lambda a: list(range(1, a+1)), newshape)

  # Indexs - I have no idea what i did :D
  ## It generates index list of splited Array
  mk = lambda arr: fmap(lambda b: apl.at(*arr[:axis]+[b]+arr[axis:]), list(range(1, apl.shape[axis]+1)))
  return APLArray(fmap(mk, flatten(indexs([], req), len(newshape) - 1)), newshape)

def Reverse(apl, axis=-1):
  if axis == -1:
    axis = len(apl.shape) - 1
  else:
    axis = axis - 1

  newshape = apl.shape
  req = fmap(lambda a: list(range(1, a+1)), apl.shape)
  req = req[:axis] + [req[axis][::-1]] + req[axis+1:] # there's gotta be a better way
  mk = lambda arr: apl.at(*arr)
  return APLArray(fmap(mk, flatten(indexs([], req), len(apl.shape) - 1)), newshape)

def ReverseFirst(apl):
  return Reverse(apl, axis=1)
  
# each L rotates each axis
# it is rotate function similar to J not Dyalog
def Rotate(l, r):
  l = APLize(l)
  r = APLize(r)
  if len(l.shape) != 1:
    raise APLError("RANK ERROR")
  if l.shape[0] > len(r.shape):
    raise APLError("RANK ERROR")
  
  rot = l.arr + [0] * (len(r.shape) - len(l.arr))
  rotate = lambda l, v: l[v:] + l[:v]
  req = zipWith(lambda a, b: rotate(list(range(1, b+1)), a), rot, r.shape)
  mk = lambda arr: r.at(*arr)
  return APLArray(fmap(mk, flatten(indexs([], req), len(r.shape) - 1)), r.shape)

def RotateFirst(l, r):
  l = APLize(l)
  r = APLize(r)
  rot = l.arr + [0] * (len(r.shape) - len(l.arr))
  return Rotate(rot[::-1], r)

def Drop(l, r):
  ## TODO check shape of left
  l = APLize(l)
  r = APLize(r)
  if len(l.shape) != 1:
    raise APLError("RANK ERROR")
  diff = fmap(lambda a: abs(a), l.arr)
  newshape = Minu(APLArray(r.shape), APLArray(diff+([0]*(len(r.shape)-len(diff)))))
  req = zipWith(lambda a, b: list(range(1 + b, a+1) if b >= 0 else range(1, a+1 + b)), r.shape, l.arr+([0]*(len(r.shape)-len(diff))))

  indxs = lambda arr, opt: list(filter(lambda a: a != [], fmap(lambda b: indxs(b, opt[1:]), fmap(lambda a: arr + [a], opt[0])))) if len(opt) > 0 else r.at(*arr)
  flatten = lambda a, d: flatten(reduce(lambda a, b: a+b, a), d-1) if d > 0 else a

  return APLArray(flatten(indxs([], req), len(r.shape) - 1), newshape.arr)

## ~ to Split, except you reduce the splited elements
def Reduce(op, apl, axis=-1):
  apl = APLize(apl)
  if len(apl.shape) == 1: # RANK 1, just reduce
    return APLArray([reduce(op, apl.arr)], [])

  if axis == -1:
    axis = len(apl.shape) - 1
  else:
    axis = axis - 1

  newshape = apl.shape[:axis] + apl.shape[axis+1:] 
  req = fmap(lambda a: list(range(1, a+1)), newshape)
  mk = lambda arr: reduce(op, fmap(lambda b: apl.at(*arr[:axis]+[b]+arr[axis:]), list(range(1, apl.shape[axis]+1))))

  return APLArray(fmap(mk, flatten(indexs([], req), len(newshape) - 1)), newshape)

# Generalized Inner Product
def dot(op1, op2):
  def f(left, right):
    left = APLize(left)
    right = APLize(right)
    # Validate : (¯1↑⍴X)≡(1↑⍴Y)
    # TODO Deal with Singleton
    if left.shape[-1] != right.shape[0]:
      raise APLError("LENGTH ERROR")

    # new shape : (¯1↓⍴X),(1↓⍴Y)
    transpos = left.shape[-1]
    newshape = left.shape[:-1] + right.shape[1:]

    if len(newshape) == 0:
      return APLArray([reduce(op1, zipWith(op2, left.arr, right.arr))], [])

    req = fmap(lambda a: list(range(1, a+1)), newshape)
    mkl = lambda arr: fmap(lambda a: left.at(*arr[:len(left.shape) - 1] + [a]), range(1, transpos+1))
    mkr = lambda arr: fmap(lambda a: right.at(*[a] + arr[len(left.shape) - 1:]), range(1, transpos+1))
    mk = lambda arr: reduce(op1, zipWith(op2, mkl(arr), mkr(arr)))

    return APLArray(fmap(mk, flatten(indexs([], req), len(newshape) - 1)), newshape)
  return f

# Outter Product
def JotDot(op):
  def f(left, right):
    left = APLize(left)
    right = APLize(right)
    if left.singleton() != None:
      return right.mapAll(lambda r: op(left.singleton(), r))
    elif right.singleton() != None:
      return left.mapAll(lambda l: op(l, right.singleton()))
    newshape = left.shape + right.shape
    req = fmap(lambda a: list(range(1, a+1)), newshape)
    mk = lambda arr: op(left.at(*arr[:len(left.shape)]), right.at(*arr[-len(right.shape):]))
    return APLArray(fmap(mk, flatten(indexs([], req), len(newshape) - 1)), newshape)
  return f

# Helper for making simple dy/monadic functions
def make_operator(d=None, m=None):
  def dyadic(left, right=None):
    # Respect Primatives
    if not isinstance(left, APLArray):
      if not isinstance(right, APLArray):
        if right != None: # Dyadic
          return d(left, right)
        else: 
          return m(left)
    # Actual APLArraies
    left = APLize(left)
    #left = left.singleton() if isinstance(left.singleton(), APLArray) else left # unwrap
    if right != None: # Dyadic
      right = APLize(right)
      #right = right.singleton() if isinstance(right.singleton(), APLArray) else right # unwrap
      if left.singleton() != None:
        if isinstance(left.singleton(), APLArray):
          return right.mapAll(lambda r: left.singleton().mapAll(lambda a: d(r, a)))
        return right.mapAll(lambda r: d(left.singleton(), r))
      elif right.singleton() != None:
        if isinstance(right.singleton(), APLArray):
          return left.mapAll(lambda l: right.singleton().mapAll(lambda a: d(l, a)))
        return left.mapAll(lambda l: d(l, right.singleton()))
      elif right.shape == left.shape:
        arr = zipWith(d, left.arr, right.arr)
        return APLArray(arr, left.shape)
      else:
        raise APLError(left, right)
    else: # Monadic
      if not m:
        raise APLError("SYNTAX ERROR: LEFT ARGUMENT REQUIRED")
      return left.mapAll(m)
  return dyadic

Iota = lambda a: APLArray(list(range(1, a+1)))

Plus = make_operator(lambda l, r: l + r               , id                                                     )
Minu = make_operator(lambda l, r: l - r               , lambda a: -1*a                                         )
Mult = make_operator(lambda l, r: l * r               , lambda a: a/abs(a)                                     )
Divi = make_operator(lambda l, r: float(l) / float(r) , lambda a: math.pow(a, -1)                              )
Star = make_operator(lambda l, r: math.pow(l, r)      , lambda a: math.pow(math.e, a)                          ) 
Min  = make_operator(lambda l, r: l if l < r else r   , lambda a: math.floor(a)                                )
Max  = make_operator(lambda l, r: l if l > r else r   , lambda a: math.ceil(a)                                 )
Magn = make_operator(lambda l, r: 0                   , lambda a: int(abs(a)) if isinstance(a, int) else abs(a))
GrE  = make_operator(lambda l, r: 1 if l >= r else 0)
Gr   = make_operator(lambda l, r: 1 if l > r else 0 )
LeE  = make_operator(lambda l, r: 1 if l <= r else 0)
Le   = make_operator(lambda l, r: 1 if l < r else 0 )
Eq   = make_operator(lambda l, r: 1 if l == r else 0)
And  = make_operator(lambda l, r: 1 if l and r else 0)
Or   = make_operator(lambda l, r: 1 if l or r else 0)

# print(Rho([2, 3, 4, 5, 6], Iota(3333))[2])
# print(Rho([2, 3,3], Iota(18)))
# print(Le(4, Rho([2, 3,3], Iota(18))))

# a = "A B C D E F G H I J K L M N O P Q R S T U V W X".split(" ")
# a = Rho([2, 3, 4], a)
# t = Rho([3, 4], Iota(1000))
# t2 = Rho([2, 2, 3, 4], Iota(1000))

# print()
# print(a)
# print(Split(a))
# print(Split(a, axis=2))

# print()
# print(t)
# print(Drop(APLArray([-1]), t))
# print(Drop(APLArray([-1, 2]), t))

# print()
# print(t2)
# print(Split(t2, axis=3))
# print(Drop(APLArray([1]), t2))
# print(Drop(APLArray([1, 1, -2]), t2))

# x = Rho([2, 2, 3], Iota(2*2*3))
# print()
# print(x)
# print(Reduce(Plus, x))

# u = Rho([2, 3, 4], Iota(10))
# i = Rho([4, 3, 2], Iota(10))
# p = Rho([4, 2], Iota(10))
# a = Rho([2, 3], Iota(6))
# b = Rho([3, 2], [10, 11, 20, 21, 30, 31])
# print()
# print(a)
# print(b)
# print(dot(Max, Mult)(u, i))

# print(dot(Plus, Mult)(Rho([2, 3], Iota(6)), Rho([3, 2], Iota(5))))

# print(dot(Plus, Mult)(APLArray([1, 2, 3]), APLArray([4, 5, 6])) == APLArray([32]))

# print(JotDot(Plus)(A([10, 20, 30]), Iota(6)))
# print(ReverseFirst(Reverse(JotDot(Plus)(A([10, 20, 30]), Iota(6)))))
# print(Reverse(JotDot(Plus)(A([10, 20, 30]), Iota(6)), axis=1))
# #print(JotDot(Plus)(u,i))
# # print(ReverseFirst(Reverse(APLArray([1, 2, 3, 4, 5]))))
# # print(APLArray([Rho([3, 3], Iota(9))]).shape)
# # print(APLArray([Rho([3, 3], Iota(9))]).singleton())
# # print(Rotate(A([1, 2]), Rho(A([3, 4]), APLArray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]))))

# # [0,0,0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,1,0,0,0,0,0,0,0]

# a = A([1, 2, 3])
# b = A([4, 5, 6])

# print("here")
# #print(life(board))
# print(type(Reduce(Plus, APLArray([1, 2, 3, 4])).singleton()))
# print(type(Reduce(Plus, A([a, a, b, b])).singleton().arr[0]))
# print(type(Reduce(Plus, A([a, a, b, b], [2, 2])).arr))

# print("===here")
# # print("there:", dot(Or, And)(APLArray([A([0, 1, 1]), A([0, 0, 1]), A([0, 1, 1]), A([0, 0, 1])], [2, 2]), APLArray([A([0, 1, 1]), A([0, 0, 1]), A([0, 1, 1]), A([0, 1, 1])], [2, 2])))
# # print("there:", dot(Or, And)(A([1, 2, 3, 4], [2, 2]), A([1, 2, 3, 4], [2, 2])))
# # print("here :", dot(Or, And)(APLArray([1, 1]), APLArray([0, 1])).arr)
# # print("here :", dot(Plus, Mult)(APLArray([1, 2]), APLArray([3, 4])))

# # print(Or(Eq(3, reduce(Plus, life(board).arr)), And(board, Eq(4, reduce(Plus, life(board).arr)))))
# # print(type(dot(Or, And)(APLArray([1, board]), APLArray([Eq(3, reduce(Plus, life(board).arr)), Eq(4, reduce(Plus, life(board).arr))])).arr[0]))

# # print(dot(Plus, Mult)(Rho([2, 3], Iota(6)), Rho([3, 2], Iota(5))))
# #print(Or(Eq(3, reduce(Plus, life(board).arr)), And(board, Eq(4, reduce(Plus, life(board).arr)))))
# #print(Eq(3, reduce(Plus, life(Rho([3,3], [1, 0, 0, 0, 1, 0, 0, 0, 0])).arr)))
# #print(Eq(4, reduce(Plus, life(Rho([3,3], [1, 0, 0, 0, 1, 0, 0, 0, 0])).arr)))

# a = APLArray([1, 2, 3])
# b = APLArray([4, 5, 6])

# print("HERE")
# print(dot(Plus, Mult)([1, 2, 3], [4, 5, 6]).singleton())
# print((dot(Plus, Mult)(APLArray([a, b]), APLArray([a, b])).singleton().shape))
# print("this", dot(Plus, Mult)(Rho([2, 3], Iota(6)), Rho([3, 2], Iota(5))))
# print("HERE")


# x = Rho([2, 2, 3], Iota(2*2*3))
# print(Split(x, axis=1))
# print(Split([1, 2]))