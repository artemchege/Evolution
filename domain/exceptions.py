class EnvironmentException(Exception):
    pass


class NotVacantPlaceException(EnvironmentException):
    """ This place is already occupied """


class UnsupportedMovement(EnvironmentException):
    """ This movement is not supported """


class ObjectNotExistsInEnvironment(EnvironmentException):
    """ Alas """


class SetupEnvironmentError(EnvironmentException):
    """ No space left in environment """


class InvalidEntityState(Exception):
    """ Invalid state """
