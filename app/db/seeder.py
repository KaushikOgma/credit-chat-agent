"""
This Module is responsible for seeding all of the master data into the DB.
"""

from app.db import get_db_instance
from app.utils.helpers.common_helper import generate_uuid
from app.utils.helpers.password_helper import hash_password
from app.utils.helpers.date_helper import get_user_time
from app.utils.constants import DBCollections
from app.utils.helpers.auth_helper import generate_api_key

class Seeder:
    """**Summary:**
    This Class is responsible for seeding all of the master data into the DB.
    """

    # CONSTANTS
    NEW_CGAI_ADMIN_USER = {
        "name": "Kaushik",
        "email": "kaushik.roy@ogmaconceptions.com",
        "password": "password",
        "contactAddress": "Kolkata",
        "contactNo": "+1234567890",
        "isActive": True,
        "isAdmin": True,
        "config": {}
    }

    # CONSTANTS
    NEW_DUMMY_USER = {
        "name": "Dummy User",
        "email": "dummy@ogmaconceptions.com",
        "password": "password",
        "contactAddress": "Kolkata",
        "contactNo": "+1234567890",
        "isActive": True,
        "isAdmin": False,
        "config": {}
    }

    async def start_seeding(self):
        """**Summary:**
        This will instantiate the object of the seeder class
        """
        # Create DB session
        db = get_db_instance()
        try:
            # Call all seeding methods
            await self.create_user(db)
        except Exception as error:
            # print("Seeder:: error - " + str(error))
            raise error
        finally:
            # Close the db connection
            db.client.close()

    async def create_user(self, db):
        """**Summary:**
        This method is responsible for inserting the master data of users,
        skipping those that already exist.

        **Args:**
            db (Session): db session reference
        """
        try:
            inserted_count = 0
            skipped_count = 0
            users_to_seed = [self.NEW_DUMMY_USER, self.NEW_CGAI_ADMIN_USER]

            for user_data in users_to_seed:
                if "id" in user_data:
                    del user_data["id"]
                user_data["password"] = await hash_password(user_data["password"])

                # Check if user exists or not
                existing_user = db[DBCollections.USER.value].find_one({"email": user_data["email"]})

                if not existing_user:
                    newUserId = generate_uuid()

                    user_data["_id"] = newUserId
                    user_data["createdAt"] = get_user_time()
                    user_data["updatedAt"] = get_user_time()
                    user_data["apiKey"] = generate_api_key()
                    db[DBCollections.USER.value].insert_one(user_data)
                    inserted_count += 1
                    print(f"User '{user_data['email']}' inserted")
                else:
                    skipped_count += 1
                    print(f"User '{user_data['email']}' already exists, skipping")

            print(
                f"Users seeding completed. Inserted: {inserted_count}, Skipped: {skipped_count}"
            )
        except Exception as error:
            print("Seeder.create_user:: error - " + str(error))
            raise error