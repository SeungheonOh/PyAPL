# -*- coding: utf-8 -*- 
import math

id = lambda a: a
fmap = lambda f, a: list(map(f, a))
reduce = lambda f, a: 0 if len(a) == 0 else f(a[0], reduce(f, a[1:])) if len(a) > 1 else a[0]
zipWith = lambda f, a, b: [f(a[0], b[0])] + zipWith(f, a[1:], b[1:]) if len(a) > 1 and len(b) > 1 else [f(a[0], b[0])]
numstr = lambda f: str(f) if not isinstance(f, float) else "{:.2f}".format(f)

# returns unicode string
def box(msg):
  lines = fmap(lambda a: a.rstrip(), msg.split("\n"))
  row = max(fmap(lambda a: len(a), lines))
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

  # shapes orders from largest demention to the least demention
  def __init__(self, arr, shape=None, box=True):
    self.box = box
    self.arr = arr
    self.shape = [len(arr)] if not shape else shape

  def __str__(self):
    longest = max(fmap(lambda a: numstr(a), self.arr), key=lambda d: len(d))
    fmt = lambda a: '{0: >{width}}'.format(numstr(a), width=len(longest))
    groupbox = id 
    if self.box:
      groupbox = box
    mkStr = lambda arr: " ".join(fmap(fmt, arr.arr)) if len(arr.shape) == 1 else groupbox("\n".join(fmap(lambda a: mkStr(arr.at(a)), range(1, arr.shape[0]+1))))
    return mkStr(self).rstrip()
  
  def __repr__(self):
    return str(self)

  # Let's make it iterable
  def __iter__(self):
    self.iterIndex = 0
    return self
  
  def next(self):
    if self.iterIndex >= self.shape[0]:
      raise StopIteration
    self.iterIndex += 1
    return self[self.iterIndex]

  def __getitem__(self, key):
    if isinstance(key, slice):
      # 1 based index
      idxs = fmap(lambda a: a+1, range(*key.indices(len(self))))
      return fmap(lambda a: self[a], idxs)
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
    #self.arr = (self.arr * need)[:need] # Wacky but whatever
    return self
  
  # Why not
  def mapAll(self, f):
    self.arr = fmap(f, self.arr)
    return self
  
  # value if singleton, None if not
  def singleton(self):
    return self.arr[0] if self.shape == [1] else None
  
  def vectorize(self):
    return self.arr

  # indexs orders from largest demention to the least demention 
  def at(self, *indexs): 
    ## Validity
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
    return APLArray(self.arr[indx:indx+amount], newshape)

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
  if right: # Dyadic
    right = APLize(right)
    if len(left.shape) == 1:
      return APLArray(right.arr, left.arr).fill()
    else:
      raise APLError("RANK ERROR", left)
  else: # Monadic
    return left.shape

def Split(apl, axis=-1):
  if axis == -1:
    axis = len(apl.shape) - 1
  else:
    axis = axis - 1
  # Range of fixed axis
  fixedRange = list(range(1, apl.shape[axis]+1))

  # axis that are not fixed
  newshape = apl.shape[:axis] + apl.shape[axis+1:]

  # Requirements
  ## list of possible indexs of all dementions except axis
  ## like [[1, 2], [1, 2, 3]]
  req = fmap(lambda a: list(range(1, a+1)), newshape)

  # Indexs - I have no idea what i did :D
  ## It generates index list of splited Array
  mk= lambda arr: fmap(lambda b: apl.at(*arr[:axis]+[b]+arr[axis:]), fixedRange)
  indxs = lambda arr, opt: fmap(lambda a: indxs(a, opt[1:]), fmap(lambda a: arr + [a], opt[0])) if len(opt) > 0 else mk(arr)

  # Reduce util...
  ## list of integer list
  flatten = lambda a, d: flatten(reduce(lambda a, b: a+b, a), d-1) if d > 0 else a

  return APLArray(flatten(indxs([], req), len(apl.shape) - 2), newshape)

# Helper for making simple dy/monadic functions
def make_operator(d, m=None):
  def dyadic(left, right=None):
    left = APLize(left)
    if right: # Dyadic
      right = APLize(right)
      if left.singleton():
        return right.mapAll(lambda r: d(left.singleton(), r))
      elif right.singleton():
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

Iota = lambda a: APLArray(range(1, a+1))

Add  = make_operator(lambda l, r: l + r, id)
Minu = make_operator(lambda l, r: l - r, lambda a: -1*a)
Mult = make_operator(lambda l, r: l * r, lambda a: a/abs(a))
Divi = make_operator(lambda l, r: float(l) / float(r), lambda a: math.pow(a, -1))
Pow  = make_operator(lambda l, r: math.pow(l, r), lambda a: math.pow(math.e, a)) 
Min  = make_operator(lambda l, r: l if l < r else r, lambda a: math.floor(a))
Max  = make_operator(lambda l, r: l if l > r else r, lambda a: math.ceil(a))
GrE  = make_operator(lambda l, r: 1 if l >= r else 0)
Gr   = make_operator(lambda l, r: 1 if l > r else 0)
LeE  = make_operator(lambda l, r: 1 if l <= r else 0)
Le   = make_operator(lambda l, r: 1 if l < r else 0)

# print(Rho([2, 3, 4, 5, 6], Iota(3333))[2])
# print(Rho([2, 3,3], Iota(18)))
# print(Le(4, Rho([2, 3,3], Iota(18))))

a = "A B C D E F G H I J K L M N O P Q R S T U V W X".split(" ")
print(a)

print(Split(Rho([3, 4], Iota(1000)), axis=1))
print(Rho([2, 3, 4], a))
print(Split(Rho([2, 3, 4], a)))
print(Split(Rho([2, 3, 4], a), axis=2))
print(Split(Rho([2, 3, 4], a), axis=1))
print(Split(Split(Rho([2, 3, 4], a))))


# test = APLArray(iota(32), [4,2,2,2])
# print("shape: " + str(test.shape))
# print(test.at(2))

# print(Minu(APLArray([1, 2, 3])))
# print(Divi(APLArray([1, 0.5, 3, 1, 0.5, 3], [2, 3])))
#print(Add(APLArray([1, 2, 3, 4], [2, 2]), APLArray([1, 2, 3, 4], [2, 2])))
# print(Pow(APLArray([1, 2, 3])))
# print(Divi(APLArray([3]), APLArray([1, 2, 3])))
# print(Max(APLArray([1, 2, 3, 4])))