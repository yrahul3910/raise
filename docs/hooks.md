# `raise_utils.hooks`

Hooks are classes that encapsulate a `Callable`, and hook before and after certain actions, as defined by the RAISE API. Because Hooks is a relatively new feature, it is not well-supported, but future development is planned. For now, only `raise_utils.hyperparams.DODGE` and `raise_utils.learners` (except `BiasedSVM` and `SVM`) support the use of hooks.

The RAISE API only provides for a base class, `Hook`. The initializer takes a name and a function as parameters, and the class provides a `call` function that forwards arguments passed to the function it encapsulates. Hooks are a way of adding user-defined behavior to standard classes.

