# `raise_utils.hooks`

Hooks are classes that encapsulate a `Callable`, and hook before and after certain actions, as defined by the RAISE API. 

Hooks are supported in various parts of the API: `raise_utils.experiments`, `raise_utils.learners`, and `raise_utils.hyperparams`

The RAISE API only provides for a base class, `Hook`. The initializer takes a name and a function as parameters, and the class provides a `call` function that forwards arguments passed to the function it encapsulates. Hooks are a way of adding user-defined behavior to standard classes.

