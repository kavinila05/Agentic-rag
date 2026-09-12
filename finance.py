from langchain_core.tools import tool


# --------------------------------------------------
# Mock financial dataset
# --------------------------------------------------

FINANCIAL_DATA = {
    "Tesla": {
        "Q2 2026": {
            "revenue": 25000,
            "revenue_unit": "million USD",
            "net_income": 1800,
            "net_income_unit": "million USD",
        }
    },

    "Industry Average": {
        "Q2 2026": {
            "revenue": 18000,
            "revenue_unit": "million USD",
            "net_income": 1200,
            "net_income_unit": "million USD",
        }
    }
}


# --------------------------------------------------
# Finance tool
# --------------------------------------------------

@tool
def finance_lookup(company: str, quarter: str) -> str:
    """
    Look up structured financial data for a company
    or industry average for a given quarter.

    Use this tool when financial numbers are required.
    """

    # Normalize common names
    company_lower = company.lower().strip()

    if company_lower in [
        "industry average",
        "automotive industry",
        "industry",
        "industry average revenue"
    ]:
        company = "Industry Average"

    elif company_lower == "tesla":
        company = "Tesla"

    company_data = FINANCIAL_DATA.get(company)

    if not company_data:
        return f"No financial data found for {company}."

    quarter_data = company_data.get(quarter)

    if not quarter_data:
        return (
            f"No financial data found for "
            f"{company} for {quarter}."
        )

    return (
        "SOURCE: MOCK_FINANCE_DATA_FOR_POC\n"
        f"Company: {company}\n"
        f"Quarter: {quarter}\n"
        f"Revenue: {quarter_data['revenue']} "
        f"{quarter_data['revenue_unit']}\n"
        f"Net income: {quarter_data['net_income']} "
        f"{quarter_data['net_income_unit']}"
    )


# --------------------------------------------------
# Direct test
# --------------------------------------------------

if __name__ == "__main__":

    result = finance_lookup.invoke(
        {
            "company": "Automotive Industry",
            "quarter": "Q2 2026"
        }
    )

    print("\n============================")
    print("FINANCE TOOL RESULT")
    print("============================")

    print(result)