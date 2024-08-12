"""
This file build a Segment Tree data structure.
    An operation for combining elements (e.g., sum, max).
    The operation must form a mathematical group together with the set of
    possible values for array elements (i.e., be associative).
    Neutral element for the operation above (e.g., float('-inf') for max and 0 for sum).
Methods
reduce(start=0, end=None)
    Returns the result of applying `self.operation`
        Beginning of the subsequence.
        End of the subsequence.
        Result of reducing self.operation over the specified range of array elements.
__setitem__(self, idx, val)
    Sets the value of an item in the array.
        Index of the item.
    val: obj
        Value to be set.
__getitem__(self, idx)
    Returns the value of an item in the array.
        Index of the item.
Builds a SumSegmentTree data structure.
Methods
sum(start=0, end=None)
    Returns the sum of a contiguous subsequence of the array.
        Beginning of the subsequence.
        End of the subsequence.
find_prefixsum_idx(self, prefixsum)
    Finds the highest index `i` in the array such that
    sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum.
    If array values are probabilities, this function
    prefixsum: np.ndarray
        Upperbound on the sum of array prefix.
        Highest index satisfying the prefixsum constraint.
Builds a MinSegmentTree data structure.
Methods
min(start=0, end=None)
    Returns the minimum value in a contiguous subsequence of the array.
        Beginning of the subsequence.
        End of the subsequence.
"""
import operator
class SegmentTree(object):
    def __init__(self, capacity, operation, neutral_element):
        """Build a Segment Tree data structure.

        https://en.wikipedia.org/wiki/Segment_tree

        Can be used as regular array, but with two
        important differences:

            a) setting item's value is slightly slower.
               It is O(lg capacity) instead of O(1).
            b) user has access to an efficient ( O(log segment size) )
               `reduce` operation which reduces `operation` over
               a contiguous subsequence of items in the array.

        Paramters
        ---------
        capacity: int
            Total size of the array - must be a power of two.
        operation: lambda obj, obj -> obj
            and operation for combining elements (eg. sum, max)
            must form a mathematical group together with the set of
            possible values for array elements (i.e. be associative)
        neutral_element: obj
            neutral element for the operation above. eg. float('-inf')
            for max and 0 for sum.
        """
        assert capacity > 0 and capacity & (capacity - 1) == 0, "capacity must be positive and a power of 2."
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation

    def _reduce_helper(self, start, end, node, node_start, node_end):
        """
        Helper function for the reduce operation.
        Recursively reduces the operation over a specified range of array elements.

        Parameters
        ----------
        start: int
            Beginning of the subsequence.
        end: int
            End of the subsequence.
        node: int
            Current node in the segment tree.
        node_start: int
            Start index of the current node's range.
        node_end: int
            End index of the current node's range.

        Returns
        -------
        reduced: obj
            Result of reducing self.operation over the specified range of array elements.
        """
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start=0, end=None):
        """Returns result of applying `self.operation`
        to a contiguous subsequence of the array.

            self.operation(arr[start], operation(arr[start+1], operation(... arr[end])))

        Parameters
        ----------
        start: int
            beginning of the subsequence
        end: int
            end of the subsequences

        Returns
        -------
        reduced: obj
            result of reducing self.operation over the specified range of array elements.
        """
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val): 
        # index of the leaf
        idx += self._capacity # 65536
        # if math.isnan(val) == True:
        #     self._value[idx] = 0
        # else:
        #     self._value[idx] = val 
            
        # if np.isnan(val).any():
        #     val = 0
        # if np.isnan(val).any():
        # print('val = ', val)
        self._value[idx] = val         
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx],
                self._value[2 * idx + 1]
            )
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity=capacity,
            operation=operator.add,
            neutral_element=0.0
        )    
    
    def sum(self, start=0, end=None) -> float:
        """Returns arr[start] + ... + arr[end]"""
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        """Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum

        if array values are probabilities, this function
        allows to sample indexes according to the discrete
        probability efficiently.

        Parameters
        ----------
        perfixsum: np.ndarray
            upperbound on the sum of array prefix

        Returns
        -------
        idx: int
            highest index satisfying the prefixsum constraint
        """
        # assert np.zeros(shape=prefixsum.shape) <= prefixsum
        # assert prefixsum <= self.sum() + 1e-5
        
        idx = 1
        while idx < self._capacity:  # while non-leaf
            if self._value[2 * idx] > prefixsum: #which is nan
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity


# Define a class MinSegmentTree that inherits from the SegmentTree class
class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        # Call the constructor of the parent class (SegmentTree)
        super(MinSegmentTree, self).__init__(
            capacity=capacity,
            operation=min,  # Set the operation to find the minimum value
            neutral_element=float('inf')  # Set the neutral element to positive infinity
        )

    def min(self, start=0, end=None):
        """Returns the minimum value in a contiguous subsequence of the array.

        Parameters
        ----------
        start: int
            Beginning of the subsequence.
        end: int
            End of the subsequence.

        Returns
        -------
        min_value: obj
            The minimum value in the specified range of array elements.
        """

        return super(MinSegmentTree, self).reduce(start, end)
