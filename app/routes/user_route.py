from fastapi import APIRouter, Request, Depends, status, Query
from typing import List, Union
from pymongo.database import Database
from app.controllers.user_controller import UserController
from fastapi.responses import JSONResponse
from starlette import status as starlette_status
from app.schemas.user_schema import UserSchemasResponse, UserSortFields, SaveUserSchema, UserDetailSchemasResponse, UserSchema, UpdateUserSchema
from app.dependencies.user_dependencies import get_user_controller
from fastapi.exceptions import HTTPException
from app.db import get_db
import datetime
from app.utils.config import settings
from app.utils.logger import setup_logger
logger = setup_logger()
router = APIRouter()

@router.get("/get_users", response_model=UserSchemasResponse)
async def get_users(
    userId: str = Query(None, description="userId"),
    name: str = Query(None, description="name"),
    email: str = Query(None, description="email"),
    startDate: str =  Query(None, description=f"startDate in {settings.ACCEPTED_DATE_TIME_STRING} format to filter createdAt"),
    endDate: str =  Query(None, description=f"endDate in {settings.ACCEPTED_DATE_TIME_STRING} format to filter createdAt"),
    sortBy: List[UserSortFields] = Query([UserSortFields.createdAt_DESC], description=f"sortBy"),
    user_controller: UserController = Depends(get_user_controller),
    db_instance: Database = Depends(get_db)
):
    """**Summary:**
    fetch all users based on filters.

    **Args:**
    - `userId` (str): Filter the users by userId. If not provided, will return all users for the logged-in user.
    - `name` (str): Filter the users by name of the user.
    - `email` (str): Filter the users by email of the request.
    - `startDate` (str): Filter the users by startDate in {settings.ACCEPTED_DATE_TIME_STRING} format.
    - `endDate` (str): Filter the users by endDate in {settings.ACCEPTED_DATE_TIME_STRING} format.
    - `sortBy` (List[RequestSortFields]): Sort the users by sortBy fields. If not provided, will sort by createdAt DESC.
    - `current_user` (User): The current user details of the logged in user.
    - `db` (Database): The database session.

    **Returns:**
    - `UserResponses`: A list of User objects containing the fetched users.
    """
    try:
        sort_params = []
        # if not current_user["isAdmin"]:
        #     userId = str(current_user["_id"])
        if startDate is not None:
            try:
                if "+" not in startDate:
                    startDate = startDate.replace(" ","+")
                # Validate the date format
                startDate = datetime.datetime.strptime(startDate, settings.ACCEPTED_DATE_TIME_STRING)
                startDate = startDate.replace(hour=0, minute=0, second=0)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"startDate must be in {settings.ACCEPTED_DATE_TIME_STRING} format")
        if endDate is not None:
            try:
                if "+" not in endDate:
                    endDate = endDate.replace(" ","+")
                # Validate the date format
                endDate = datetime.datetime.strptime(endDate, settings.ACCEPTED_DATE_TIME_STRING)
                endDate = endDate.replace(hour=23, minute=59, second=59)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"endDate must be in {settings.ACCEPTED_DATE_TIME_STRING} format")
        # Validate sort parameters
        if sortBy is not None and len(sortBy) > 0:
            for item in sortBy:
                [field, order] = item.value.split(":")
                if field not in [field for (field, _) in sort_params]:
                    sort_params.append((field, 1 if order == "ASC" else -1))
            # Convert the list of tuples into a dictionary
            sort_params = {field: order for field, order in sort_params}
        async with db_instance as db:
            data = await user_controller.get_users(db, userId, startDate, endDate, name, email, sort_params)
            return JSONResponse(
                status_code=200, content={"data": data, "message": "Data fetched successfully"}
            )
    except Exception as error:
        logger.exception(error)
        return JSONResponse(content={"message": str(error)}, status_code=500)



@router.get("/get_user_details/{use_id}", response_model=UserDetailSchemasResponse)
async def get_user_details(
    user_id: str,
    user_controller: UserController = Depends(get_user_controller),
    db_instance: Database = Depends(get_db)
):
    """Fetch the current user's details.

    **Args:**
    - `current_user` (User): The current user details of the logged in user.
    - `db` (Database): Dependency to get the database session.

    **Returns:**
    - `UserDetails`: The current user's details.
    """
    try:
        # Fetch the current user's details from the database
        async with db_instance as db:
            data = await user_controller.get_user_detail(db, user_id)
            return JSONResponse(
                status_code=200, content={"data": data, "message": "Data fetched successfully"}
            )
    except Exception as error:
        # Log the error and return a JSON response with the error message
        logger.exception(error)
        return JSONResponse(content={"message": str(error)}, status_code=500)



@router.post("/add_user", status_code=status.HTTP_201_CREATED)
async def add_user(
    body: SaveUserSchema,
    user_controller: UserController = Depends(get_user_controller),
    db_instance: Database = Depends(get_db)
):
    """**Summary:**
    Add a new user to the system.

    **Args:**
    - `body` (User): Request body containing the user details to be added.
    - `current_user` (User): Current user details of the logged-in user.
    - `db` (Database): Dependency to get the database session.

    **Returns:**
    - A response containing the newly added user details.
    """
    try:
        # Convert the request body to a dictionary
        user_data = body.model_dump()

        # Call the add_user method of the user controller
        async with db_instance as db:
            data = await user_controller.add_user(user_data, db)
            return JSONResponse(
                        status_code=200, content={"data": data, "message": "Data added successfully"}
                    )
    except Exception as error:
        # Log the error
        logger.exception(error)
        # Return a JSON response with an error message
        return JSONResponse(content={"message": str(error)}, status_code=500)



@router.post("/update_user/{id}", status_code=status.HTTP_201_CREATED)
async def update_user(
    id: str,
    body: UpdateUserSchema,
    user_controller: UserController = Depends(get_user_controller),
    db_instance: Database = Depends(get_db)
):
    """**Summary:**
    Update an existing user in the system.

    **Args:**
    - `id` (int): The ID of the user to be updated.
    - `body` (User): The request body containing the user details to be updated.
    - `current_user` (User, optional): The current user details of the logged-in user. Defaults to the result of the `get_current_user` dependency.
    - `db` (Database, optional): The dependency to get the database session. Defaults to the result of the `get_db` dependency.

    **Returns:**
    - The updated user details.
    """
    try:
        # Convert the request body to a dictionary
        user_data = body.model_dump(exclude_unset=True)

        # Call the add_user method of the user controller
        async with db_instance as db:
            update_flag = await user_controller.update_user(id, user_data, db)
            if update_flag:
                return JSONResponse(
                            status_code=200, content={"message": "Data updated successfully"}
                        )
            else:
                return JSONResponse(
                            status_code=400, content={"message": "Invalid request"}
                        )
    except Exception as error:
        # Log the error
        logger.exception(error)
        # Return a JSON response with an error message
        return JSONResponse(content={"message": str(error)}, status_code=500)
