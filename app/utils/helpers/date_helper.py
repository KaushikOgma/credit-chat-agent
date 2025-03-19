from datetime import timezone
import datetime
import os
import pytz
from dateutil import parser
import re
from app.utils.config import settings


def get_utc_time(to_string=True):
    """**Summary:**
    Generate and return the current UTC datetime as a string.

    **Args:**
    - `to_string` (Boolean): if true then it will return datetime as string otherwise at dattime object
    """
    try:
        current_utc_time = datetime.datetime.now(timezone.utc)
        if to_string:
            formatted_utc_time = current_utc_time.strftime(settings.ACCEPTED_DATE_TIME_STRING)
        else:
            formatted_utc_time = current_utc_time
        return formatted_utc_time
    except Exception as error:
        # print("get_utc_time:: error - " + str(error))
        raise error

def get_user_time(to_string=False, timeZone = None):
    """**Summary:**
    Generate and return the current User datetime as a string.

    **Args:**
    - `to_string` (Boolean): if true then it will return datetime as string otherwise at dattime object
    """
    try:
        if timeZone is None:
            user_timezone = pytz.timezone(settings.APP_TIMEZONE)
            current_user_time = datetime.datetime.now(user_timezone)
        else:
            current_timezone = pytz.timezone(timeZone)
            current_user_time = datetime.datetime.now(current_timezone)

        if to_string:
            formatted_user_time = current_user_time.strftime(settings.ACCEPTED_DATE_TIME_STRING)
        else:
            formatted_user_time = current_user_time
        return formatted_user_time
    except Exception as error:
        # print("get_user_time:: error - " + str(error))
        raise error

def get_time_from_iso(time, to_string=False, timeZone=None):
    """
    **Summary:**
    Generate and return the current User datetime as a string based on a provided date.

    **Args:**
    - `time` (str): ISO date string to be used for generating the user time.
    - `to_string` (Boolean): If true, then it will return datetime as a string; otherwise, as a datetime object.
    - `timeZone` (str, optional): The timezone to use. Defaults to None.

    **Returns:**
    - Formatted user time as a string if `to_string` is True; otherwise, a datetime object.
    """
    try:
        # Ensure ISO format string is correct
        match = re.match(r"(.*)([+-]\d{2})(\d{2})$", time)
        if match:
            time = f"{match.group(1)}{match.group(2)}:{match.group(3)}"

        # Parse the provided ISO date string
        provided_time = parser.isoparse(time)

        if timeZone is None:
            user_timezone = pytz.timezone(settings.APP_TIMEZONE)
        else:
            user_timezone = pytz.timezone(timeZone)

        # Localize the provided time to the user timezone
        localized_user_time = provided_time.astimezone(user_timezone)

        if to_string:
            formatted_user_time = localized_user_time.strftime(settings.ACCEPTED_DATE_TIME_STRING)
        else:
            formatted_user_time = localized_user_time

        return formatted_user_time
    except Exception as error:
        print("get_user_time:: error - " + str(error))
        raise error

def convert_timezone(date, to_string=False, timeZone = None):
    """**Summary:**
    Generate and return the current User datetime as a string.

    **Args:**
    - `to_string` (Boolean): if true then it will return datetime as string otherwise at dattime object
    """
    try:
        if timeZone is None:
            user_timezone = pytz.timezone(settings.APP_TIMEZONE)
            date_converted = date.astimezone(user_timezone)
        else:
            if isinstance(timeZone,str):
                current_timezone = pytz.timezone(timeZone)
            else:
                current_timezone = timeZone
            date_converted = date.astimezone(current_timezone)

        if to_string:
            formatted_user_time = date_converted.strftime(settings.ACCEPTED_DATE_TIME_STRING)
        else:
            formatted_user_time = date_converted
        return formatted_user_time
    except Exception as error:
        # print("convert_timezone:: error - " + str(error))
        raise error

