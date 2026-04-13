from pydantic import BaseModel, Field, field_validator
from typing import Literal
import json

class UserDatabaseBaseAPI(BaseModel) :
    success : bool
    user_id : str 
    user_email : str 
    status : Literal['Retiree', 'Employee', 'Ex-Employee']
    prioritize : bool

    @field_validator('user_email')
    @classmethod
    def validate_email(cls, v:str) -> str:
        if "@" not in v or "." not in v :
            raise ValueError(f"{v} is not a valid email")
        return v


class SupportTicket(BaseModel):
    title : str = Field(description="Concise title of ticket")
    user_id : str = Field(description="User id of customer if available", frozen=True)
    user_name : str = Field(description="Name of customer")
    ticket_category : Literal['Technical', 'HR', 'General']
    ticket_priority : Literal['Low', 'Medium', 'High']
    ticket_summary : str = Field(min_length = 10, max_length= 1000, description= "Should be descriptive but concise, preferably within a paragraph")

userInfo = UserDatabaseBaseAPI (
    success=True,
    user_id="kit_kat",
    user_email="kitkat@gmail.com",
    status="Employee",
    prioritize=True
)

ticket = SupportTicket(
    title="Customer did not receieve last month's pay",
    user_id="kit_kat",
    user_name="Kit Katherine",
    ticket_category="HR",
    ticket_priority="High",
    ticket_summary="User stated that despite expecting a higher pay after a promotion he has not seen any payment for last month."
)

# print(userInfo.model_dump())
print(userInfo.model_dump_json())


