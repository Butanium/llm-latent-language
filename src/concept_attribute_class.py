from attr import field, define 
from typing import List, Union

@define
class ConceptAttribute:
    """
    a concept is represented by attributes for a specific relation
    
    """
    concept: str = field()
    attrs: List[Union[List, str]] = field()
    freqs: List[int] = field() # len(freqs) == len(attrs)
    relation: str = field(default=None)
    
    n_attrs: int = field(default=None)
    
    def __attrs_post_init__(self):
        if len(self.attrs) != len(self.freqs):
            raise ValueError(f'Ensure attributes (attrs) has same length as frequencies (freqs)\
                                -- current lengths: len(attrs)={len(self.attrs)}, len(freqs)={len(self.freqs)}')
            
        self.n_attrs = len(self.attrs)
    
    def compare(self, other_attrs) -> float: 
        """ 
        Take a different set of attributes 
        Check to see how many of them are equal. 
        
        Return: n_equal / n_attrs
        """
        if isinstance(other_attrs, ConceptAttribute):
            other_attrs = other_attrs.attrs  
        equal = False
        count_equal = 0
        for attr in self.attrs: 
            for other_attr in other_attrs: 
                equal = self._check_equality_of_features(attr, other_attr)
                if equal: 
                    count_equal += 1 
                    break 
            equal = False
            
        return count_equal / self.n_attrs
    
    
    @staticmethod
    def _check_equality_of_features(f1: Union[List, str], f2: Union[List, str]) -> bool: 
        # compare two list elements to see if they are equal
        if isinstance(f1, str): 
            if isinstance(f2, str):
                return f1 == f2 
            else:
                return f1 in f2 
        else: 
            if isinstance(f2, str):
                return f2 in f1 
            else:
                contains = 0 
                for f in f1:
                    if f in f2:
                        contains += 1 
                return bool(contains)