import math

id = lambda a: a
fmap = lambda f, a: list(map(f, a))
reduce = lambda f, a: 0 if len(a) == 0 else f(a[0], reduce(f, a[1:])) if len(a) > 1 else a[0]
zipWith = lambda f, a, b: [f(a[0], b[0])] + zipWith(f, a[1:], b[1:]) if len(a) > 1 and len(b) > 1 else [f(a[0], b[0])]
numstr = lambda f: str(f) if not isinstance(f, float) else "{:.2f}".format(f)

iota = lambda a: range(1, a+1)

class APLError(Exception):
  pass

class APLArray:
  ## NOTE
  # self.arr is ALWAYS a vector
  # self.shape will determine the dementions of data

  # shapes orders from largest demention to the least demention
  def __init__(self, arr, shape=None):
    self.arr = arr
    self.shape = [len(arr)] if not shape else shape

  def __str__(self):
    longest = max(fmap(lambda a: numstr(a), self.arr), key=lambda d: len(d))
    fmt = lambda a: '{0: <{width}}'.format(numstr(a), width=len(longest))
    mkStr = lambda arr: " ".join(fmap(fmt, arr.arr)) if len(arr.shape) == 1 else "\n".join(fmap(lambda a: mkStr(arr.at(a)), iota(arr.shape[0]))) + "\n"
    return mkStr(self).rstrip()
  
  # Fill empty spaces to fufill shape
  def fill(self): 
    need = reduce(lambda a, b: a*b, self.shape)
    if need < len(self.arr): # good
      return
    self.arr = (self.arr * need)[:need] # Wacky but whatever
    return self
  
  # Why not
  def map(self, f):
    self.arr = fmap(f, self.arr)
    return self
  
  # value if singleton, None if not
  def singleton(self):
    return self.arr[0] if self.shape == [1] else None

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
                          , iota(len(self.shape)-1)) + [1]

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

# Helper for making simple dy/monadic functions
def make_operator(m, d):
  def dyadic(left, right=None):
    left = APLize(left)
    if right: # Dyadic
      right = APLize(right)
      if left.singleton():
        return right.map(lambda r: d(left.singleton(), r))
      elif right.singleton():
        return left.map(lambda l: d(l, right.singleton()))
      elif right.shape == left.shape:
        arr = zipWith(d, left.arr, right.arr)
        return APLArray(arr, left.shape)
      else:
        raise APLError(left, right)
    else: # Monadic
      return left.map(m)
  return dyadic

Add  = make_operator(id, lambda l, r: l + r)
Minu = make_operator(lambda a: -1*a, lambda l, r: l - r)
Mult = make_operator(lambda a: a/abs(a), lambda l, r: l * r)
Divi = make_operator(lambda a: math.pow(a, -1), lambda l, r: float(l) / float(r))
Pow  = make_operator(lambda a: math.pow(math.e, a),lambda l, r: math.pow(l, r)) 
Min  = make_operator(lambda a: math.floor(a), lambda l, r: l if l < r else r)
Max  = make_operator(lambda a: math.ceil(a), lambda l, r: l if l > r else r)

print(Rho(40, 1))

# test = APLArray(iota(32), [4,2,2,2])
# print("shape: " + str(test.shape))
# print(test.at(2))

# print(Minu(APLArray([1, 2, 3])))
# print(Divi(APLArray([1, 0.5, 3, 1, 0.5, 3], [2, 3])))
#print(Add(APLArray([1, 2, 3, 4], [2, 2]), APLArray([1, 2, 3, 4], [2, 2])))
# print(Pow(APLArray([1, 2, 3])))
# print(Divi(APLArray([3]), APLArray([1, 2, 3])))
# print(Max(APLArray([1, 2, 3, 4])))
