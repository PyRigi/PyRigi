(input-checks)=
# Input Checks

To unify to a certain extent error messages there are two ways.

## Check the input before actual computations
For checking input type and values for standard data types
we collected a set of methods that are often needed in ```_input_check.py```.
For instance ```_input_check.dimension(dim)``` checks whether
the parameter ```dim``` is a positive integer and raises an error otherwise.

Checks which need properties of the calling object
are in the respective class of the object and
have a name starting with ```_input_check_```.
For instance ```G._input_check_no_loop()``` checks whether
a graph is loop free and raises an error otherwise.

Note that these input checks are considered private methods.
However, we do test them.


## Check the input at the end
In some cases a method would run a conditional statement
through all possible options of an input parameter.
If the method reaches the end
an error would be raised to tell the user that the option is not supported.
For instance a parameter ```algorithm``` might have several supported values.
When the methods reaches its end in an else condition an
```NotSupportedValueError(algorithm, "algorithm", self.method) ``` is raised.
