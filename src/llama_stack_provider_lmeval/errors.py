# Custom exceptions
class LMEvalError(Exception):
    """Base exception for LMEval errors"""

    pass


class LMEvalConfigError(LMEvalError):
    """Configuration related errors"""

    pass


class LMEvalValidationError(LMEvalError):
    """Validation related errors"""

    pass
