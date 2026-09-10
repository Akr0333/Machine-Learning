"""Secure password generator CLI."""
import argparse
import secrets
import string


def generate_password(length: int = 16, symbols: bool = True) -> str:
    """Generate a cryptographically secure random password."""
    if length < 4:
        raise ValueError("Password length must be at least 4")
    alphabet = string.ascii_letters + string.digits
    if symbols:
        alphabet += "!@#$%^&*()-_=+"
    return "".join(secrets.choice(alphabet) for _ in range(length))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a secure random password")
    parser.add_argument("-l", "--length", type=int, default=16)
    parser.add_argument("--no-symbols", action="store_true")
    args = parser.parse_args()
    print(generate_password(args.length, not args.no_symbols))


if __name__ == "__main__":
    main()
