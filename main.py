from dotenv import load_dotenv

from graph.graph import app

load_dotenv()

if __name__ == "__main__":
    print(
        app.invoke(
            {
                "question": "¿Quiénes están obligados al cumplimiento de las disposiciones aduaneras?"
            }
        ).get("generation")
    )
