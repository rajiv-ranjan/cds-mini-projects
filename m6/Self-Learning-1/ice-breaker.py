import os
from dotenv import dotenv_values, load_dotenv

def load_configs_and_secrets():
    load_dotenv(dotenv_path=".env-secret")
    
    config = {
        **dotenv_values(".env"),  # load sensitive variables
        **dotenv_values(".env-shared"),  # load shared variables
        # **os.environ,  # override loaded values with environment variables
    }
    return config

def main():
    # read the documentation at https://pypi.org/project/python-dotenv/
    # I'm loading the secret in the os environment and other configs from .env and .env-shared

    config = load_configs_and_secrets()
    
    print(f"from .env: {config['MESSAGE']}")
    print(f"from .env-shared: {config['HOST']}")
    print(f"from .env-secret: {os.environ['API_KEY']}")


if __name__ == "__main__":
    main()
