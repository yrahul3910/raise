"""
Scott-Knot test + non parametric effect size + significance tests.
Tim Menzies, 2019. Share and enjoy. No warranty. Caveat Emptor.

Accepts data as per the following exmaple (you can ignore the "*n"
stuff, that is just there for the purposes of demos on larger
and larger data)

Ouputs treatments, clustered such that things that have similar
results get the same ranks.
"""
from copy import deepcopy as kopy


class o:
    def __init__(self, **d): self.__dict__.update(**d)


class THE:
    cliffs = o(dull=[0.147,  # small
                     0.33,  # medium
                     0.474  # large
                     ])
    bs = o(conf=0.05,
           b=500)
    mine = o(private="_")
    char = o(skip="?")
    rx = o(show="%4s %10s %s")
    tile = o(width=50,
             chops=[0.1, 0.25, 0.5, 0.75, 0.9],
             marks=[" ", "-", "-", "-", " "],
             bar="|",
             star="*",
             show=" %5.3f")


# -----------------------------------------------------
def cliffsDelta(lst1, lst2, dull=THE.cliffs.dull[0]):
    "By pre-soring the lists, this cliffsDelta runs in NlogN time"

    def runs(lst):
        for j, two in enumerate(lst):
            if j == 0:
                one, i = two, 0
            if one != two:
                yield j - i, one
                i = j
            one = two
        yield j - i + 1, two

    # ---------------------
    m, n = len(lst1), len(lst2)
    lst2 = sorted(lst2)
    j = more = less = 0
    for repeats, x in runs(sorted(lst1)):
        while j <= (n - 1) and lst2[j] < x:
            j += 1
        more += j * repeats
        while j <= (n - 1) and lst2[j] == x:
            j += 1
        less += (n - j) * repeats
    d = (more - less) / (m * n)
    return abs(d) <= dull


def same(x): return x


class Mine:
    "class that, amongst other times, pretty prints objects"
    oid = 0

    def identify(self):
        Mine.oid += 1
        self.oid = Mine.oid
        return self.oid


class Rx(Mine):
    "place to manage pairs of (TreatmentName,ListofResults)"

    def __init__(self, rx="", vals=None, dull=THE.cliffs.dull[0]):
        if vals is None:
            vals = []
        self.rx = rx
        self.vals = sorted([x for x in vals if x != THE.char.skip])
        self.n = len(self.vals)
        self.med = self.vals[int(self.n / 2)]
        self.mu = sum(self.vals) / self.n
        self.rank = 1

    def tiles(self, lo=0, hi=1):
        return xtile(self.vals, lo, hi)

    def __lt__(self, other):
        return self.med < other.med

    def __eq__(self, other):
        return cliffsDelta(self.vals, other.vals, dull=self.dull)  # and \

    def __repr__(self):
        return '%4s %10s %s' % (self.rank, self.rx, self.tiles())

    def xpect(self, j, b4):
        "Expected value of difference in means before and after a split"
        n = self.n + j.n
        return self.n / n * (b4.med - self.med) ** 2 + j.n / n * (j.med - b4.med) ** 2

    # -- end instance methods --------------------------

    @staticmethod
    def data(**d):
        "convert dictionary to list of treatments"
        return [Rx(k, v) for k, v in d.items()]

    @staticmethod
    def sum(rxs):
        "make a new rx from all the rxs' vals"
        all_ = []
        for rx in rxs:
            for val in rx.vals:
                all_ += [val]
        return Rx(vals=all_)

    @staticmethod
    def show(rxs):
        "pretty print set of treatments"
        for rx in sorted(rxs):
            print(THE.rx.show % (rx.rank, rx.rx, rx.tiles()))

    @staticmethod
    def sk(rxs, effect='small'):
        "sort treatments and rank them"
        effect_dict = {
            'small': THE.cliffs.dull[0],
            'medium': THE.cliffs.dull[1],
            'large': THE.cliffs.dull[2]
        }

        if effect in effect_dict:
            effect = effect_dict[effect]

        def divide(lo, hi, b4, rank):
            cut = left = right = None
            best = 0
            for j in range(lo + 1, hi):
                left0 = Rx.sum(rxs[lo:j])
                right0 = Rx.sum(rxs[j:hi])
                now = left0.xpect(right0, b4)
                if now > best:
                    if not cliffsDelta(left0.vals, right0.vals, dull=effect):
                        best, cut, left, right = now, j, kopy(
                            left0), kopy(right0)
            if cut:
                rank = divide(lo, cut, left, rank) + 1
                rank = divide(cut, hi, right, rank)
            else:
                for rx in rxs[lo:hi]:
                    rx.rank = rank
            return rank

        # -- sk main
        rxs = sorted(rxs)
        divide(0, len(rxs), Rx.sum(rxs), 1)
        return rxs


# -------------------------------------------------------
def pairs(lst):
    "Return all pairs of items i,i+1 from a list."
    last = lst[0]
    for i in lst[1:]:
        yield last, i
        last = i


def words(f):
    with open(f) as fp:
        for line in fp:
            for word in line.split(', '):
                yield word


def xtile(lst, lo, hi,
          width=THE.tile.width,
          chops=THE.tile.chops,
          marks=THE.tile.marks,
          bar=THE.tile.bar,
          star=THE.tile.star,
          show=THE.tile.show):
    """The function _xtile_ takes a list of (possibly)
    unsorted numbers and presents them as a horizontal
    xtile chart (in ascii format). The default is a
    contracted _quintile_ that shows the
    10,30,50,70,90 breaks in the data (but this can be
    changed- see the optional flags of the function).
    """

    def pos(p):
        return ordered[int(len(lst) * p)]

    def place(x):
        return int(width * float((x - lo)) / (hi - lo + 0.00001))

    def pretty(lst):
        return ', '.join([show % x for x in lst])

    ordered = sorted(lst)
    lo = min(lo, ordered[0])
    hi = max(hi, ordered[-1])
    what = [pos(p) for p in chops]
    where = [place(n) for n in what]
    out = [" "] * width
    for one, two in pairs(where):
        for i in range(one, two):
            out[i] = marks[0]
        marks = marks[1:]
    out[int(width / 2)] = bar
    out[place(pos(0.5))] = star
    return '(' + ''.join(out) + ")," + pretty(what)


def thing(x):
    "Numbers become numbers; every other x is a symbol."
    try:
        return int(x)
    except ValueError:
        try:
            return float(x)
        except ValueError:
            return x
