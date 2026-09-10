import pytest
from password_generator import generate_password


def test_password_length():
    assert len(generate_password(20)) == 20


def test_symbols_can_be_disabled():
    password = generate_password(30, symbols=False)
    assert all(char.isalnum() for char in password)


def test_invalid_length():
    with pytest.raises(ValueError):
        generate_password(3)
