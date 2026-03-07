"""Custom exceptions for the application."""


class AppError(Exception):
    def __init__(self, message: str, details: str = ""):
        self.message = message
        self.details = details
        super().__init__(self.message)


class RepositoryCloneError(AppError):
    pass


class LLMError(AppError):
    pass


class LLMResponseValidationError(LLMError):
    pass