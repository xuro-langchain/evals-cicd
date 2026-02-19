"""Tools for the multi-agent system."""

import sqlite3
import requests
from langchain_core.tools import tool
from langchain_community.utilities.sql_database import SQLDatabase
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool


def get_engine_for_chinook_db():
    """Load Chinook sample database into memory."""
    url = "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"
    response = requests.get(url)
    sql_script = response.text

    connection = sqlite3.connect(":memory:", check_same_thread=False)
    connection.executescript(sql_script)
    return create_engine(
        "sqlite://",
        creator=lambda: connection,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )


engine = get_engine_for_chinook_db()
db = SQLDatabase(engine)


# Invoice Tools
@tool
def get_invoices_by_customer(customer_id: str) -> str:
    """Get all invoices for a customer, sorted by date (most recent first).

    Args:
        customer_id: The customer's ID number
    """
    return db.run(
        f"SELECT * FROM Invoice WHERE CustomerId = {customer_id} ORDER BY InvoiceDate DESC LIMIT 5;"
    )


@tool
def get_invoice_total(customer_id: str) -> str:
    """Get the total amount spent by a customer across all invoices.

    Args:
        customer_id: The customer's ID number
    """
    return db.run(
        f"SELECT SUM(Total) as TotalSpent FROM Invoice WHERE CustomerId = {customer_id};"
    )


invoice_tools = [get_invoices_by_customer, get_invoice_total]


# Music Catalog Tools
@tool
def get_albums_by_artist(artist: str) -> str:
    """Search for albums by an artist name.

    Args:
        artist: The artist name to search for (partial match supported)
    """
    return db.run(
        f"""
        SELECT Album.Title, Artist.Name
        FROM Album
        JOIN Artist ON Album.ArtistId = Artist.ArtistId
        WHERE Artist.Name LIKE '%{artist}%'
        LIMIT 10;
        """,
        include_columns=True,
    )


@tool
def get_tracks_by_artist(artist: str) -> str:
    """Get songs/tracks by an artist.

    Args:
        artist: The artist name to search for
    """
    return db.run(
        f"""
        SELECT Track.Name as Song, Album.Title as Album, Artist.Name as Artist
        FROM Track
        JOIN Album ON Track.AlbumId = Album.AlbumId
        JOIN Artist ON Album.ArtistId = Artist.ArtistId
        WHERE Artist.Name LIKE '%{artist}%'
        LIMIT 10;
        """,
        include_columns=True,
    )


@tool
def search_tracks(query: str) -> str:
    """Search for tracks by name.

    Args:
        query: The track name to search for
    """
    return db.run(
        f"""
        SELECT Track.Name as Song, Artist.Name as Artist, Album.Title as Album
        FROM Track
        JOIN Album ON Track.AlbumId = Album.AlbumId
        JOIN Artist ON Album.ArtistId = Artist.ArtistId
        WHERE Track.Name LIKE '%{query}%'
        LIMIT 10;
        """,
        include_columns=True,
    )


music_tools = [get_albums_by_artist, get_tracks_by_artist, search_tracks]
