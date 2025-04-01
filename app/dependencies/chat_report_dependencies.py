
from app.repositories.credit_report_repositories import CreditReportRepository
from app.services.credit_report_extractor import CreditReportExtractor
from app.services.credit_report_processor import CreditReportProcessor
from app.controllers.credit_report_controller import CreditReportController

def get_credit_report_controller():
    """
    Returns an instance of CreditReportController.

    Returns:
        CreditReportController: An instance of CreditReportController.
    """
    credit_report_repo = CreditReportRepository()
    credit_report_extractor = CreditReportExtractor()
    credit_report_processor = CreditReportProcessor()
    return CreditReportController(credit_report_extractor, credit_report_processor, credit_report_repo)

# Export the required function
__all__ = [
    "get_credit_report_controller",
]