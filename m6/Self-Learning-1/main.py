import os
from dotenv import load_dotenv


def main():
    load_dotenv(dotenv_path=".env")
    load_dotenv(dotenv_path=".env-shared")

    print("from .env: " + os.getenv("MESSAGE"))
    print("from .env-shared: " + os.getenv("HOST"))


if __name__ == "__main__":
    main()
