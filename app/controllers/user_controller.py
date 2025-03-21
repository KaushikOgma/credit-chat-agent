from fastapi.responses import JSONResponse
from app.utils.helpers.date_helper import get_user_time, convert_timezone
from app.repositories.user_repositories import UserRepository
from datetime import datetime
from app.utils.config import settings
from fastapi.exceptions import HTTPException
from app.utils.logger import setup_logger
logger = setup_logger()


class UserController:

    def __init__(self, user_repo: UserRepository):
        self.user_repo = user_repo
        self.service_name = "user_manage_service"

    async def get_users(
        self,
        db,
        userId: str,
        startDate: datetime,
        endDate: datetime,
        name: str,
        email: str,
        company: str,
        sort_params: list,
    ) -> dict:
        """
        **Summary:** This method is responsible for fetching all users based on the provided criteria.
        **Args:**
        - `db` (Database): db session referance.
        - `userId` (str): comma separated list of user Ids to filter by.
        - `startDate` (datetime): start date of the time range for which data is to be retrieved.
        - `endDate` (datetime): end date of the time range for which data is to be retrieved.
        - `name` (str): filter the users by name.
        - `email` (str): filter the users by email.
        - `company` (str): filter the users by company.
        - `sort_params` (list): sorting parameters for the data.
        **Returns:**
        - A dictionary containing the list of users and a success message.
        """
        try:
            filterData = {}
            input_timezone = None
            if userId is not None:
                filterData["_id"] = {
                        '$in': [int(id) for id in userId.split(",")]
                    }
            if name is not None:
                filterData["name"] = name
            if email is not None:
                filterData["email"] = email
            if company is not None:
                filterData["company"] = company
            if startDate is not None:
                input_timezone = startDate.tzname().replace("UTC","")
                filterData["createdAt"] = {
                    '$gte': convert_timezone(startDate, to_string=False, timeZone="UTC"),
                }
                if endDate is None:
                    endDate = startDate.replace(hour=23, minute=59, second=59)
                    filterData["createdAt"]["$lte"] = convert_timezone(endDate, to_string=False, timeZone="UTC")
                else:
                    filterData["createdAt"]["$lte"] = convert_timezone(endDate, to_string=False, timeZone="UTC")
            data = await self.user_repo.get_users(db, filterData, sort_params, input_timezone)
            # Convert datetime objects to strings if they're not already strings
            for user in data:
                if user.get('createdAt'):
                    if isinstance(user['createdAt'], datetime):
                        user['createdAt'] = user['createdAt'].strftime(settings.ACCEPTED_DATE_TIME_STRING)
                if user.get('updatedAt'):
                    if isinstance(user['updatedAt'], datetime):
                        user['updatedAt'] = user['updatedAt'].strftime(settings.ACCEPTED_DATE_TIME_STRING)
            return JSONResponse(
                        status_code=200, content={"data": data, "message": "Data fetched successfully"}
                    )
        except Exception as error:
            logger.exception(error)
            raise error
    

    async def get_user_detail(self, db, userId):
        """**Summary:**
        This method is responsible for fetching all users.

        **Args:**
        - `db` (Database): db session referance.
        - `userId` (int): Id of the user for which need to retrive the details.
        """
        try:
            data = await self.user_repo.get_user_details_by_id(db, userId)
            return JSONResponse(
                status_code=200, content={"data": data, "message": "Data fetched successfully"}
            )
        except Exception as error:
            logger.exception(error)
            raise error


    async def add_user(self, user_data, db):
        """**Summary:**
        This method is responsible for adding a user with an API key.

        **Args:**
        - `user_data` (Dict): user data to be inserted.
        - `db` (Database): db session referance.
        """
        try:
            inserted_id = await self.user_repo.add_user(db, user_data)
            return JSONResponse(
                        status_code=200, content={"id":inserted_id, "message": "Data inserted successfully"}
                    )
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            raise error


    async def update_user(self, id, user_data, db):
        """**summary**
        A method to update an existing user based on the provided user id and user data.

        **Args:**
        - `id` (int): The user id to update.
        - `user_data` (Dict): The user data to update.
        - `db` (Database): The database session reference.

        **Returns:**
        - A message dict indicating the success of the update operation.
        - If the user id is invalid, returns a JSONResponse with a 401 status code and a message indicating an invalid user id.
        """
        try:
            # print("id:: ",id)
            # print("user_data:: ",user_data)
            # Check if order exists or not
            update_flag = await self.user_repo.update_user(db, id, user_data)

            if update_flag:
                return JSONResponse(
                    status_code=200, content={"message": "Data updated successfully"}
                )
            else:
                return JSONResponse(
                    status_code=400, content={"message": "Invalid user id"}
                )
        except Exception as error:
            logger.exception(error)
            raise error