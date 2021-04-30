def piter(x, percent_period=1, period=None, end="| ", show=True):
    """Iterates through x (any iterable object, having len) returning iteratively elements from x and printing progress.
    Progress is printed every <period> iterations or after every <percent_period> percent of total was complete.
    Useful for controlling how much of the long computations were completed.
    Example:
        for i in piter([10,11,12,13,14,15],2):
            print(i)
    Output:
        0.00% done
        10
        11
        33.33% done
        12
        13
        66.67% done
        14
        15
    Author: Victor Kitov (v.v.kitov@yandex.ru), 03.2016."""

    if show == False:  # do nothing
        for element in x:
            yield element
    else:

        if hasattr(x, "__len__"):
            for i, element in enumerate(x):
                yield element
        else:
            for i, element in enumerate(x):
                yield element


class Struct:
    """
    Structure data type.
    Examples of use:
    A=cStruct()
    A.property1=value1
    A.property2=value2
    ...
    A=cStruct(property1=value1,property2=value2,...)
    """

    def __init__(self, **keywords):
        self._fields = []
        for sKey in list(keywords.keys()):
            setattr(self, sKey, keywords[sKey])

    def get_str(self, sSeparator):
        sString = ""
        lsAttributes = list(vars(self).keys())

        for sAttribute in lsAttributes:
            if sAttribute[0] != "_":
                attr = getattr(self, sAttribute)
                if isinstance(attr, (np.ndarray, pd.Series, pd.DataFrame)):
                    sString += f"{sAttribute}=...,{sSeparator}"
                else:
                    sAttributeValue = str(getattr(self, sAttribute))
                    if len(sAttributeValue) > 30:
                        sString += f"{sAttribute}=...,{sSeparator}"
                    else:
                        sString += (
                            f"{sAttribute}={sAttributeValue},{sSeparator}"
                        )
        return "{" + sString[:-2] + "}"

    @property
    def pstr(self):
        return self.get_str("\n")

    def __str__(self):
        return self.get_str(" ")

    def __repr__(self):
        return self.get_str(" ")

    def __unicode__(self):
        return self.get_str(" ")

    def __eq__(self, other):
        return (
            self.GetAttributes2ValuesDict() == other.GetAttributes2ValuesDict()
        )

    def __ne__(self, other):
        return (
            self.GetAttributes2ValuesDict() != other.GetAttributes2ValuesDict()
        )

    def __hash__(self):
        return hash(self.GetAttributes2ValuesTuple())

    def get_defaults(self, oDefaultStruct):
        """
        A=cStruct(i=1,j=2,k=3)
        B=cStruct(a=10,b=20,c=30,i=333)
        A.GetDefaults(B)
        print A
        Structure: a=10 c=30 b=20 i=1 k=3 j=2"""
        lsDefaultAttributes = list(vars(oDefaultStruct).keys())
        for sDefaultAttribute in lsDefaultAttributes:
            if not hasattr(self, sDefaultAttribute):
                setattr(
                    self,
                    sDefaultAttribute,
                    getattr(oDefaultStruct, sDefaultAttribute),
                )

    @property
    def fields(self):
        return self._fields

    def get_fields(self):
        return self._fields

    @property
    def fields2values(self):
        dAttributes2Values = {}
        for sAttribute in list(vars(self).keys()):
            dAttributes2Values[sAttribute] = getattr(self, sAttribute)
        return dAttributes2Values

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name != "_fields":
            if name not in self._fields:
                self._fields.append(name)

    def __delattr__(self, name):
        super().__delattr__(name)
        if name != "_fields":
            if name in self._fields:
                self._fields.remove(name)
