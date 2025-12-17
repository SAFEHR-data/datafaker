Working with CSV and Parquet files with DuckDB
==============================================

DuckDB is supported for both the source and the destination database.
DuckDB is famous for being able to work with ``.parquet`` and ``.csv`` files without first pulling the data into its own database.
(A Parquet file is a data format like CSV but much more efficient for storage and searching, and not human readable).

Using DuckDB without using Parquet or CSV files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

DuckDB will work as source and/or destination databases like this:

.. code-block:: shell

   export SRC_DSN=duckdb:////path/to/file/duck.db
   export DST_DSN=duckdb:////path/to/file/fake.db

This will use the DuckDB database in the file ``/path/to/file/duck.db`` and output to the file ``/path/to/file/fake.db``.
The destination file must exist before the call to ``datafaker create-tables``.
This is achieved most simply like this:

.. code-block:: shell

   duckdb /path/to/file/fake.db -c ""

This will write an empty database to the file ``/path/to/file/fake.db``.

Using DuckDB to source from Parquet or CSV files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You do not need to set up a real DuckDB database if all your source data is in Parquet or CSV files.
Simply set your source connection string to an in-memory database:

.. code-block:: shell

   export SRC_DSN=duckdb:///:memory:

If you do this, `datafaker create-tables` will not work as there is nothing in this database, so you will have to write your ``orm.yaml`` file by hand.
Assume you have two Parquet files, ``artist.parquet`` and ``artwork.parquet``.

Our ``orm.yaml`` file must specify these files and the columns within them, for example:

.. code-block:: yaml

   tables:
      artist.parquet:  # this is the name of the parquet file
         columns:
            artist_id:
               type: INTEGER
               primary: true  # mark artist_id as the primary key
               nullable: false  # columns are nullable by default, so set this if not.
            name:
               type: TEXT
            gender:
               type: TEXT
            nationality:
               type: TEXT
            birth_date:
               type: DATE
            end_date:
               type: DATE
      artwork.parquet:  # The other parquet file
         columns:
            artwork_id:
               type: INTEGER
               primary: true
               nullable: false
            artist_id:
               foreign_keys:
               - artist.parquet.artist_id  # Maps to the artist_id column of the artist.parquet file
            name:
               type: TEXT
            date:
               type: date
            medium:
               type: TEXT

Now you can run ``datafaker configure-tables`` and similar to begin configuring datafaker.

Using DuckDB to write fake Parquet or CSV files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You cannot use an in-memory DuckDB for the destination database because it needs to survive multiple calls to ``datafaker``.
Luckily it is not hard to set up a DuckDB database; use the command above (``duckdb fake.db -c ""``) and set your ``DST_DSN`` string accordingly.

Now you can use ``datafaker create-tables``, ``datafaker create-generators``, and ``datafaker create-data``.
You now have a file containing the fake data. If you want CSV of parquet files you can use the following commands:

.. code-block:: shell

   mkdir fake_csv
   datafaker dump-data --output fake_csv/
   mkdir fake_parquet
   datafaker dump-data --parquet --output fake_parquet/

You now have a directory of fake data in CSV format and anther directory of the same fake data in parquet format.
