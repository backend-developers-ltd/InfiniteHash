import enum

from django.db import connection


class LockType(enum.Enum):
    WEIGHT_SETTING = 1
    VALIDATION_SCHEDULING = 2


class Locked(Exception):
    pass


def get_advisory_lock(type_: LockType) -> None:
    """
    Obtain postgres advisory lock.
    Has to be executed in transaction.atomic context. Throws `Locked` if not able to obtain the lock. The lock
    will be released automatically after transaction.atomic ends.
    """
    cursor = connection.cursor()
    cursor.execute("SELECT pg_try_advisory_xact_lock(%s)", [type_.value])
    unlocked = cursor.fetchall()[0][0]
    if not unlocked:
        raise Locked
