from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import research_report as report


class MarkdownLinkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "research" / "claims" / "claim.md"
        self.source.parent.mkdir(parents=True)
        self.generated = self.root / "research" / "generated"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def validate(
        self, text: str, *, require_generated_links: bool = False
    ) -> tuple[list[str], list[str]]:
        self.source.write_text(text, encoding="utf-8")
        return report.validate_markdown_links(
            self.root,
            self.source,
            text,
            self.generated,
            require_generated_links=require_generated_links,
        )

    def test_existing_local_and_external_links_pass(self) -> None:
        target = self.root / "evidence" / "result.md"
        target.parent.mkdir()
        target.write_text("result\n", encoding="utf-8")

        errors, warnings = self.validate(
            "[local](../../evidence/result.md), [reference][evidence] and "
            "[external](https://example.org)\n\n"
            "[evidence]: ../../evidence/result.md\n"
        )

        self.assertEqual(errors, [])
        self.assertEqual(warnings, [])

    def test_missing_local_link_is_an_error(self) -> None:
        errors, warnings = self.validate("[missing](../../evidence/missing.md)")

        self.assertEqual(len(errors), 1)
        self.assertEqual(warnings, [])

    def test_generated_link_is_warning_before_export_and_error_after(self) -> None:
        text = "[html](../generated/html/report.html)"

        errors, warnings = self.validate(text)
        strict_errors, strict_warnings = self.validate(
            text, require_generated_links=True
        )

        self.assertEqual(errors, [])
        self.assertEqual(len(warnings), 1)
        self.assertEqual(len(strict_errors), 1)
        self.assertEqual(strict_warnings, [])

    def test_links_in_code_are_ignored(self) -> None:
        errors, warnings = self.validate(
            "`[inline](missing.md)`\n\n```text\n[block](missing.md)\n```\n"
        )

        self.assertEqual(errors, [])
        self.assertEqual(warnings, [])


class ArchiveTransactionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.archive = Path(self.temporary.name) / "archive"
        self.archive.mkdir()
        self.manifest = self.archive / ".research-mirror-manifest.json"
        self.manifest.write_text(
            json.dumps({"schema_version": 1, "files": {}}, indent=2) + "\n",
            encoding="utf-8",
        )
        self.target = self.archive / "docs" / "result.md"
        self.target.parent.mkdir()
        self.target.write_bytes(b"old\n")
        self.old_hash = report.sha256(self.target)
        self.new_hash = hashlib.sha256(b"new\n").hexdigest()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def prepare_transaction(self) -> tuple[Path, dict[str, object]]:
        transaction_id = "a" * 32
        transaction = self.archive / ".research-mirror-staging" / transaction_id
        incoming = transaction / "incoming" / "docs" / "result.md"
        incoming.parent.mkdir(parents=True)
        incoming.write_bytes(b"new\n")
        journal: dict[str, object] = {
            "schema_version": 1,
            "transaction_id": transaction_id,
            "old_manifest_sha256": report.sha256(self.manifest),
            "replacements": [
                {
                    "relative": "docs/result.md",
                    "had_original": True,
                    "old_sha256": self.old_hash,
                    "new_sha256": self.new_hash,
                }
            ],
            "archives": [],
        }
        report.write_json_atomic(transaction / "journal.json", journal)
        return transaction, journal

    def test_manifest_failure_rolls_back_previous_copy(self) -> None:
        transaction, journal = self.prepare_transaction()
        manifest_value = {
            "schema_version": 1,
            "transaction_id": transaction.name,
            "files": {"docs/result.md": {"sha256": self.new_hash}},
        }
        original_writer = report.write_json_atomic

        def fail_manifest(path: Path, value: object) -> None:
            if path == self.manifest:
                raise OSError("injected manifest failure")
            original_writer(path, value)

        with mock.patch.object(report, "write_json_atomic", side_effect=fail_manifest):
            with self.assertRaises(OSError):
                report.publish_archive_transaction(
                    self.archive,
                    self.manifest,
                    transaction,
                    journal,
                    {"docs/result.md": {"sha256": self.new_hash}},
                    manifest_value,
                )

        self.assertEqual(self.target.read_bytes(), b"old\n")
        self.assertFalse(transaction.exists())
        self.assertNotIn(
            "transaction_id",
            json.loads(self.manifest.read_text(encoding="utf-8")),
        )

    def test_next_run_recovers_interrupted_replacement(self) -> None:
        transaction, _ = self.prepare_transaction()
        backup = transaction / "backup" / "docs" / "result.md"
        backup.parent.mkdir(parents=True)
        os.replace(self.target, backup)
        os.replace(transaction / "incoming" / "docs" / "result.md", self.target)

        report.recover_archive_transactions(
            self.archive, self.manifest, dry_run=False
        )

        self.assertEqual(self.target.read_bytes(), b"old\n")
        self.assertFalse(transaction.exists())
        self.assertFalse((self.archive / ".research-mirror-staging").exists())

    def test_manifest_failure_restores_archived_file(self) -> None:
        stale = self.archive / "obsolete.md"
        stale.write_bytes(b"obsolete\n")
        stale_hash = report.sha256(stale)
        transaction = self.archive / ".research-mirror-staging" / ("b" * 32)
        transaction.mkdir(parents=True)
        archived = self.archive / "_removed" / transaction.name / "obsolete.md"
        journal = {
            "schema_version": 1,
            "transaction_id": transaction.name,
            "old_manifest_sha256": report.sha256(self.manifest),
            "replacements": [],
            "archives": [
                {
                    "source": "obsolete.md",
                    "target": archived.relative_to(self.archive).as_posix(),
                    "sha256": stale_hash,
                }
            ],
        }
        report.write_json_atomic(transaction / "journal.json", journal)

        with mock.patch.object(
            report, "write_json_atomic", side_effect=OSError("injected failure")
        ):
            with self.assertRaises(OSError):
                report.publish_archive_transaction(
                    self.archive,
                    self.manifest,
                    transaction,
                    journal,
                    {},
                    {"schema_version": 1, "transaction_id": transaction.name},
                )

        self.assertEqual(stale.read_bytes(), b"obsolete\n")
        self.assertFalse(archived.exists())

    def test_committed_transaction_is_cleaned_without_rollback(self) -> None:
        transaction, _ = self.prepare_transaction()
        backup = transaction / "backup" / "docs" / "result.md"
        backup.parent.mkdir(parents=True)
        os.replace(self.target, backup)
        os.replace(transaction / "incoming" / "docs" / "result.md", self.target)
        archived = self.archive / "_removed" / transaction.name / "obsolete.md"
        archived.parent.mkdir(parents=True)
        archived.write_bytes(b"obsolete\n")
        archived_hash = report.sha256(archived)
        self.manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "transaction_id": transaction.name,
                    "files": {
                        "docs/result.md": {"sha256": self.new_hash}
                    },
                    "archived_removed": [
                        {
                            "source": "obsolete.md",
                            "archive": archived.relative_to(self.archive).as_posix(),
                            "sha256": archived_hash,
                        }
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (transaction / "journal.json").unlink()

        report.recover_archive_transactions(
            self.archive, self.manifest, dry_run=False
        )

        self.assertEqual(self.target.read_bytes(), b"new\n")
        self.assertEqual(archived.read_bytes(), b"obsolete\n")
        self.assertFalse(transaction.exists())

    def test_next_run_cleans_interrupted_staging_before_journal(self) -> None:
        transaction = self.archive / ".research-mirror-staging" / ("c" * 32)
        incoming = transaction / "incoming" / "docs" / "result.md"
        incoming.parent.mkdir(parents=True)
        incoming.write_bytes(b"new\n")
        report.write_json_atomic(
            transaction / "state.json",
            {
                "schema_version": 1,
                "transaction_id": transaction.name,
                "phase": "staging",
            },
        )

        report.recover_archive_transactions(
            self.archive, self.manifest, dry_run=False
        )

        self.assertEqual(self.target.read_bytes(), b"old\n")
        self.assertFalse(transaction.exists())


if __name__ == "__main__":
    unittest.main()
