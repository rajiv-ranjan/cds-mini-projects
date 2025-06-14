import os
from dotenv import load_dotenv


def main():
    load_dotenv()
    print(os.getenv("HOST"))
    print("Hello from self-learning-1!")


if __name__ == "__main__":
    main()
