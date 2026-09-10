"""
    Standalone Qdrant diagnostic script.

    Purpose:
        Inspect an existing local Qdrant database and identify:

        1. Total physical points
        2. Unique logical (document_id, chunk_id) pairs
        3. Duplicate logical chunks
        4. Missing document_id / chunk_id
        5. Identical content stored multiple times
        6. Documents with their chunk counts
        7. Suspicious / malformed points

    IMPORTANT:
        This script is READ-ONLY.
        It does NOT delete, update, or modify Qdrant data.

    Run from your project root:

        python diagnose_qdrant.py

    Expected database:

        ./qdrant_db

    Expected collection:

        documents
"""

from collections import Counter, defaultdict
from pathlib import Path
import sys

from qdrant_client import QdrantClient


# ============================================================
# CONFIGURATION
# ============================================================

QDRANT_PATH = "./qdrant_db"
COLLECTION_NAME = "documents"

# Number of problematic examples to print.
SAMPLE_LIMIT = 20

# Scroll batch size.
BATCH_SIZE = 256


# ============================================================
# HELPERS
# ============================================================

def normalize(value):
    """
    Normalize metadata values for comparison.

    None, empty string, and missing values are treated as missing.
    """
    if value is None:
        return None

    value = str(value).strip()

    if not value:
        return None

    return value


def get_metadata(point):
    """
    Safely extract metadata from a Qdrant point.
    """
    payload = point.payload or {}

    metadata = payload.get("metadata")

    if not isinstance(metadata, dict):
        metadata = {}

    return metadata


def get_content(point):
    """
    Safely extract stored text content.
    """
    payload = point.payload or {}

    content = payload.get("text", "")

    if content is None:
        return ""

    return str(content)


# ============================================================
# MAIN DIAGNOSTIC
# ============================================================

def main():

    print("=" * 80)
    print("QDRANT DATABASE DIAGNOSTIC")
    print("=" * 80)

    print(f"\nQdrant path : {Path(QDRANT_PATH).resolve()}")
    print(f"Collection  : {COLLECTION_NAME}")

    # --------------------------------------------------------
    # Verify database path
    # --------------------------------------------------------

    qdrant_path = Path(QDRANT_PATH)

    if not qdrant_path.exists():
        print("\nERROR: Qdrant database path does not exist:")
        print(qdrant_path.resolve())
        sys.exit(1)

    # --------------------------------------------------------
    # Connect
    # --------------------------------------------------------

    print("\nConnecting to Qdrant...")

    try:
        client = QdrantClient(path=QDRANT_PATH)

    except Exception as exc:
        print("\nERROR: Could not connect to Qdrant.")
        print(exc)
        sys.exit(1)

    # --------------------------------------------------------
    # Verify collection
    # --------------------------------------------------------

    try:
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]

    except Exception as exc:
        print("\nERROR: Could not retrieve collections.")
        print(exc)
        sys.exit(1)

    print("\nAvailable collections:")

    for name in collection_names:
        print(f"  - {name}")

    if COLLECTION_NAME not in collection_names:
        print(
            f"\nERROR: Collection '{COLLECTION_NAME}' "
            "does not exist."
        )
        sys.exit(1)

    # --------------------------------------------------------
    # Collection information
    # --------------------------------------------------------

    try:
        collection_info = client.get_collection(
            collection_name=COLLECTION_NAME
        )

    except Exception as exc:
        print("\nERROR: Could not inspect collection.")
        print(exc)
        sys.exit(1)

    print("\n" + "-" * 80)
    print("COLLECTION INFORMATION")
    print("-" * 80)

    print(
        f"Points reported by Qdrant: "
        f"{collection_info.points_count}"
    )

    # ========================================================
    # STORAGE STRUCTURES
    # ========================================================

    total_points = 0

    logical_chunks = Counter()

    document_chunks = defaultdict(set)

    content_occurrences = Counter()

    content_metadata = defaultdict(set)

    missing_document_id = []
    missing_chunk_id = []
    missing_both = []

    malformed_metadata = []

    all_points = []

    # ========================================================
    # SCAN QDRANT
    # ========================================================

    print("\nScanning points...")

    offset = None

    while True:

        try:
            points, next_offset = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=BATCH_SIZE,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

        except Exception as exc:
            print("\nERROR while scanning Qdrant:")
            print(exc)
            sys.exit(1)

        if not points:
            break

        for point in points:

            total_points += 1

            metadata = get_metadata(point)

            document_id = normalize(
                metadata.get("document_id")
            )

            chunk_id = normalize(
                metadata.get("chunk_id")
            )

            source = normalize(
                metadata.get("source")
            )

            content = get_content(point)

            # ------------------------------------------------
            # Store point information
            # ------------------------------------------------

            point_info = {
                "point_id": str(point.id),
                "document_id": document_id,
                "chunk_id": chunk_id,
                "source": source,
                "content": content,
            }

            all_points.append(point_info)

            # ------------------------------------------------
            # Missing metadata
            # ------------------------------------------------

            if document_id is None:
                missing_document_id.append(point_info)

            if chunk_id is None:
                missing_chunk_id.append(point_info)

            if document_id is None and chunk_id is None:
                missing_both.append(point_info)

            # ------------------------------------------------
            # Malformed metadata
            # ------------------------------------------------

            if not isinstance(point.payload, dict):
                malformed_metadata.append(point_info)

            # ------------------------------------------------
            # Logical chunk tracking
            # ------------------------------------------------

            if document_id is not None and chunk_id is not None:

                logical_key = (
                    document_id,
                    chunk_id
                )

                logical_chunks[logical_key] += 1

                document_chunks[document_id].add(
                    chunk_id
                )

            # ------------------------------------------------
            # Content tracking
            # ------------------------------------------------

            if content:

                content_occurrences[content] += 1

                metadata_key = (
                    document_id,
                    chunk_id
                )

                content_metadata[content].add(
                    metadata_key
                )

        offset = next_offset

        if offset is None:
            break

    # ========================================================
    # BASIC STATISTICS
    # ========================================================

    unique_logical_chunks = len(logical_chunks)

    duplicate_logical_chunks = {
        key: count
        for key, count in logical_chunks.items()
        if count > 1
    }

    duplicate_physical_points = sum(
        count - 1
        for count in logical_chunks.values()
        if count > 1
    )

    duplicate_content = {
        content: count
        for content, count in content_occurrences.items()
        if count > 1
    }

    identical_content_different_metadata = {}

    for content, metadata_set in content_metadata.items():

        if len(metadata_set) > 1:

            identical_content_different_metadata[
                content
            ] = metadata_set

    # ========================================================
    # REPORT
    # ========================================================

    print("\n" + "=" * 80)
    print("1. BASIC COUNTS")
    print("=" * 80)

    print(f"Physical Qdrant points        : {total_points}")
    print(f"Unique logical chunks        : {unique_logical_chunks}")
    print(
        f"Duplicate physical points   : "
        f"{duplicate_physical_points}"
    )

    if total_points > 0:
        duplicate_percentage = (
            duplicate_physical_points
            / total_points
            * 100
        )
    else:
        duplicate_percentage = 0.0

    print(
        f"Duplicate point percentage   : "
        f"{duplicate_percentage:.2f}%"
    )

    print("\n" + "=" * 80)
    print("2. METADATA INTEGRITY")
    print("=" * 80)

    print(
        f"Missing document_id          : "
        f"{len(missing_document_id)}"
    )

    print(
        f"Missing chunk_id             : "
        f"{len(missing_chunk_id)}"
    )

    print(
        f"Missing BOTH IDs             : "
        f"{len(missing_both)}"
    )

    print(
        f"Malformed payload            : "
        f"{len(malformed_metadata)}"
    )

    print("\n" + "=" * 80)
    print("3. DOCUMENT STATISTICS")
    print("=" * 80)

    print(
        f"Documents with valid IDs     : "
        f"{len(document_chunks)}"
    )

    if document_chunks:

        for document_id in sorted(document_chunks):

            chunks = document_chunks[document_id]

            print(
                f"  {document_id} "
                f"-> {len(chunks)} unique chunks"
            )

    # ========================================================
    # DUPLICATE LOGICAL CHUNKS
    # ========================================================

    print("\n" + "=" * 80)
    print("4. DUPLICATE LOGICAL CHUNKS")
    print("=" * 80)

    if not duplicate_logical_chunks:

        print("No duplicate (document_id, chunk_id) pairs found.")

    else:

        print(
            f"Found {len(duplicate_logical_chunks)} "
            "logical chunks stored more than once.\n"
        )

        for index, (key, count) in enumerate(
            sorted(
                duplicate_logical_chunks.items(),
                key=lambda item: item[1],
                reverse=True
            ),
            start=1
        ):

            document_id, chunk_id = key

            print(
                f"{index}. "
                f"document_id={document_id} | "
                f"chunk_id={chunk_id} | "
                f"physical_points={count}"
            )

            if index >= SAMPLE_LIMIT:
                remaining = (
                    len(duplicate_logical_chunks)
                    - SAMPLE_LIMIT
                )

                if remaining > 0:
                    print(
                        f"\n... and {remaining} more."
                    )

                break

    # ========================================================
    # MISSING METADATA
    # ========================================================

    print("\n" + "=" * 80)
    print("5. POINTS WITH MISSING METADATA")
    print("=" * 80)

    if not missing_both:

        print("No points with both IDs missing.")

    else:

        print(
            f"Found {len(missing_both)} points "
            "with BOTH document_id and chunk_id missing.\n"
        )

        for index, point in enumerate(
            missing_both[:SAMPLE_LIMIT],
            start=1
        ):

            content_preview = (
                point["content"]
                .replace("\n", " ")
                .strip()
            )

            if len(content_preview) > 150:
                content_preview = (
                    content_preview[:150] + "..."
                )

            print(
                f"{index}. "
                f"point_id={point['point_id']}"
            )

            print(
                f"   source={point['source']}"
            )

            print(
                f"   content={content_preview}"
            )

    # ========================================================
    # PARTIAL METADATA
    # ========================================================

    print("\n" + "=" * 80)
    print("6. PARTIALLY MISSING METADATA")
    print("=" * 80)

    partial_metadata = [
        point
        for point in all_points
        if (
            (
                point["document_id"] is None
                and point["chunk_id"] is not None
            )
            or
            (
                point["document_id"] is not None
                and point["chunk_id"] is None
            )
        )
    ]

    print(
        f"Points with only one ID present: "
        f"{len(partial_metadata)}"
    )

    for index, point in enumerate(
        partial_metadata[:SAMPLE_LIMIT],
        start=1
    ):

        print(
            f"{index}. "
            f"point_id={point['point_id']} | "
            f"document_id={point['document_id']} | "
            f"chunk_id={point['chunk_id']} | "
            f"source={point['source']}"
        )

    # ========================================================
    # DUPLICATE CONTENT
    # ========================================================

    print("\n" + "=" * 80)
    print("7. DUPLICATE CONTENT")
    print("=" * 80)

    print(f"Unique contents appearing more than once: {len(duplicate_content)}")

    if duplicate_content:

        for index, (content, count) in enumerate(
            sorted(
                duplicate_content.items(),
                key=lambda item: item[1],
                reverse=True
            )[:SAMPLE_LIMIT],
            start=1
        ):

            preview = (content.replace("\n", " ").strip())

            if len(preview) > 150:
                preview = preview[:150] + "..."

            print(f"\n{index}. occurrences={count}")

            print(f"   content={preview}")


    # SAME CONTENT + DIFFERENT METADATA
    print("\n" + "=" * 80)
    print("8. IDENTICAL CONTENT WITH DIFFERENT METADATA")
    print("=" * 80)

    print("This is especially useful for investigating the None-ID retrieval problem.")

    print(
        f"\nContents associated with multiple metadata "
        f"combinations: "
        f"{len(identical_content_different_metadata)}"
    )

    if identical_content_different_metadata:

        for index, (content,metadata_set) in enumerate(
            identical_content_different_metadata.items(),
            start=1):

            preview = (content.replace("\n", " ").strip())

            if len(preview) > 150:
                preview = preview[:150] + "..."

            print(f"\n{index}. metadata variants:")

            for metadata in metadata_set:
                print(f"   {metadata}")

            print(f"   content={preview}")

            if index >= SAMPLE_LIMIT:
                break


    # POINTS FOR A SPECIFIC LOGICAL CHUNK
    print("\n" + "=" * 80)
    print("9. DUPLICATE POINT DETAILS")
    print("=" * 80)

    if duplicate_logical_chunks:

        printed = 0

        for (document_id,chunk_id) in duplicate_logical_chunks:

            print(f"\nLogical chunk:")

            print(f"  document_id={document_id}")

            print(f"  chunk_id={chunk_id}")

            matching_points = [
                point
                for point in all_points
                if (
                    point["document_id"] == document_id
                    and point["chunk_id"] == chunk_id
                )
            ]

            for point in matching_points:
                print(f" point_id={point['point_id']} | "f"source={point['source']}")

            printed += 1

            if printed >= SAMPLE_LIMIT:
                break

    else:
        print("No duplicate logical chunks to inspect.")


    # Final Diagnosis
    print("\n" + "=" * 80)
    print("10. DIAGNOSTIC SUMMARY")
    print("=" * 80)

    problems_found = False

    if duplicate_logical_chunks:
        problems_found = True
        print("\n[PROBLEM] Duplicate logical chunks detected.")
        print("The same (document_id, chunk_id) exists as multiple physical Qdrant points.")
        print("This can cause duplicate chunks to consume retrieval top-K positions.")

    if missing_both:
        problems_found = True
        print("\n[PROBLEM] Points with missing document_id AND chunk_id detected.")
        print("These points cannot participate correctly in document/chunk-level evaluation.")

    if partial_metadata:
        problems_found = True
        print("\n[PROBLEM] Points with partially missing metadata detected.")

    if identical_content_different_metadata:
        problems_found = True
        print("\n[PROBLEM] Identical content exists with different metadata.")
        print("This is particularly suspicious when one metadata variant contains None IDs.")

    if not problems_found:
        print("\nNo obvious duplicate or metadata-integrity problems were detected.")
        print("If retrieval is still returning incorrect "
            "chunks, the next investigation should focus "
            "on embedding/retrieval behavior rather than "
            "index duplication.")

    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)

    client.close()


# Entry Point
if __name__ == "__main__":
    main()