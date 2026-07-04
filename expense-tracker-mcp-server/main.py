from fastmcp import FastMCP
import os
import json
import sqlite3
import contextlib
from datetime import date as date_cls

DB_PATH = os.path.join(os.path.dirname(__file__), "expenses.db")
CATEGORIES_PATH = os.path.join(os.path.dirname(__file__), "categories.json")

mcp = FastMCP("ExpenseTracker")


def get_connection():
    """
    Single place that opens a DB connection.
    WAL mode + busy_timeout let multiple short-lived processes (e.g. a fresh
    MCP subprocess per tool call from a chatbot) read/write the same file
    without throwing 'database is locked' under light concurrent access.
    """
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    return conn


def init_db():
    with contextlib.closing(get_connection()) as c:
        with c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS expenses(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    amount REAL NOT NULL,
                    category TEXT NOT NULL,
                    subcategory TEXT DEFAULT '',
                    note TEXT DEFAULT ''
                )
            """)


init_db()


# ---------- helpers ----------

def load_categories():
    """Read categories.json fresh each time so it can be edited without restarting."""
    with open(CATEGORIES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_category(category, subcategory=""):
    """
    Normalize + validate category/subcategory against categories.json.
    Raises ValueError with a helpful message if invalid.
    Returns (normalized_category, normalized_subcategory).
    """
    if not category or not category.strip():
        raise ValueError("Category is required.")

    cats = load_categories()
    cat_key = category.strip().lower().replace(" ", "_")

    if cat_key not in cats:
        raise ValueError(
            f"Unknown category '{category}'. Valid categories: {sorted(cats.keys())}"
        )

    sub_key = ""
    if subcategory and subcategory.strip():
        sub_key = subcategory.strip().lower().replace(" ", "_")
        if sub_key not in cats[cat_key]:
            raise ValueError(
                f"Unknown subcategory '{subcategory}' for category '{cat_key}'. "
                f"Valid subcategories: {cats[cat_key]}"
            )

    return cat_key, sub_key


def validate_date(date_str):
    """Ensure date is a valid ISO (YYYY-MM-DD) string."""
    try:
        date_cls.fromisoformat(date_str)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid date '{date_str}'. Expected format: YYYY-MM-DD.")
    return date_str


def validate_amount(amount):
    """Ensure amount is a positive number."""
    try:
        amount = float(amount)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid amount '{amount}'. Must be a number.")
    if amount <= 0:
        raise ValueError(f"Amount must be greater than 0, got {amount}.")
    return amount


# ---------- tools ----------

@mcp.tool()
def get_categories():
    """
    Return the full list of allowed categories and subcategories.
    Always call this before add_expense/update_expense to pick a valid
    category/subcategory rather than inventing new ones.
    """
    return load_categories()


@mcp.tool()
def add_expense(date, amount, category, subcategory="", note=""):
    '''Add a new expense entry to the database.
    category and subcategory must match values from get_categories().'''
    try:
        date = validate_date(date)
        amount = validate_amount(amount)
        category, subcategory = validate_category(category, subcategory)
    except ValueError as e:
        return {"status": "error", "message": str(e)}

    with contextlib.closing(get_connection()) as c:
        with c:
            cur = c.execute(
                "INSERT INTO expenses(date, amount, category, subcategory, note) VALUES (?,?,?,?,?)",
                (date, amount, category, subcategory, note)
            )
            return {"status": "ok", "id": cur.lastrowid}


@mcp.tool()
def list_expenses(start_date, end_date):
    '''List expense entries within an inclusive date range.'''
    try:
        start_date = validate_date(start_date)
        end_date = validate_date(end_date)
    except ValueError as e:
        return {"status": "error", "message": str(e)}

    with contextlib.closing(get_connection()) as c:
        cur = c.execute(
            """
            SELECT id, date, amount, category, subcategory, note
            FROM expenses
            WHERE date BETWEEN ? AND ?
            ORDER BY id ASC
            """,
            (start_date, end_date)
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


@mcp.tool()
def summarize(start_date, end_date, category=None):
    '''Summarize expenses by category within an inclusive date range.'''
    try:
        start_date = validate_date(start_date)
        end_date = validate_date(end_date)

        cat_key = None
        if category:
            cat_key, _ = validate_category(category)
    except ValueError as e:
        return {"status": "error", "message": str(e)}

    with contextlib.closing(get_connection()) as c:
        query = (
            """
            SELECT category, SUM(amount) AS total_amount
            FROM expenses
            WHERE date BETWEEN ? AND ?
            """
        )
        params = [start_date, end_date]

        if cat_key:
            query += " AND category = ?"
            params.append(cat_key)

        query += " GROUP BY category ORDER BY category ASC"

        cur = c.execute(query, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


@mcp.tool()
def list_all_expenses():
    """Return all expenses stored in the database."""
    with contextlib.closing(get_connection()) as c:
        cur = c.execute("""
            SELECT id, date, amount, category, subcategory, note
            FROM expenses
            ORDER BY id ASC
        """)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


@mcp.tool()
def delete_expense(expense_id: int):
    """
    Delete an expense by its ID.
    """
    with contextlib.closing(get_connection()) as c:
        with c:
            cur = c.execute(
                "DELETE FROM expenses WHERE id = ?",
                (expense_id,)
            )

            if cur.rowcount == 0:
                return {
                    "status": "error",
                    "message": f"No expense found with ID {expense_id}"
                }

            return {
                "status": "success",
                "message": f"Expense {expense_id} deleted successfully"
            }


@mcp.tool()
def update_expense(
    expense_id: int,
    date: str | None = None,
    amount: float | None = None,
    category: str | None = None,
    subcategory: str | None = None,
    note: str | None = None
):
    """
    Update one or more fields of an existing expense.
    category/subcategory (if provided) must match values from get_categories().
    """
    updates = []
    values = []

    try:
        if date is not None:
            date = validate_date(date)
            updates.append("date = ?")
            values.append(date)

        if amount is not None:
            amount = validate_amount(amount)
            updates.append("amount = ?")
            values.append(amount)

        # category/subcategory need to be validated together since subcategory
        # depends on category. If only one is given, look up the current value
        # of the other from the DB so validation is still correct.
        if category is not None or subcategory is not None:
            with contextlib.closing(get_connection()) as c:
                cur = c.execute(
                    "SELECT category, subcategory FROM expenses WHERE id = ?",
                    (expense_id,)
                )
                row = cur.fetchone()
                if row is None:
                    return {
                        "status": "error",
                        "message": f"No expense found with ID {expense_id}"
                    }
                current_category, current_subcategory = row

            effective_category = category if category is not None else current_category
            effective_subcategory = subcategory if subcategory is not None else current_subcategory

            effective_category, effective_subcategory = validate_category(
                effective_category, effective_subcategory
            )

            if category is not None:
                updates.append("category = ?")
                values.append(effective_category)
            if subcategory is not None:
                updates.append("subcategory = ?")
                values.append(effective_subcategory)

        if note is not None:
            updates.append("note = ?")
            values.append(note)
    except ValueError as e:
        return {"status": "error", "message": str(e)}

    if not updates:
        return {
            "status": "error",
            "message": "No fields provided for update"
        }

    values.append(expense_id)

    query = f"""
        UPDATE expenses
        SET {', '.join(updates)}
        WHERE id = ?
    """

    with contextlib.closing(get_connection()) as c:
        with c:
            cur = c.execute(query, values)

            if cur.rowcount == 0:
                return {
                    "status": "error",
                    "message": f"No expense found with ID {expense_id}"
                }

            return {
                "status": "success",
                "message": f"Expense {expense_id} updated successfully"
            }


@mcp.resource("expense://categories", mime_type="application/json")
def categories():
    # Kept for clients that do support MCP resources; get_categories() tool
    # above is the one that's guaranteed to work everywhere.
    with open(CATEGORIES_PATH, "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    mcp.run()
