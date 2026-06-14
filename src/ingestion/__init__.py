from src.ingestion.diabetes_loader import DiabetesLoader
from src.ingestion.home_credit_loader import HomeCreditLoader
from src.ingestion.acs_income_loader import ACSIncomeLoader

LOADER_MAP = {
    "diabetes_130us": DiabetesLoader,
    "home_credit":    HomeCreditLoader,
    "acs_income":     ACSIncomeLoader,
}