"""Tests for the post-hoc import grounding check."""

from agents.grounding import check_import_grounding


class TestImportGrounding:
    def test_python_stdlib_is_grounded(self):
        report = check_import_grounding(
            "import os\nimport json\nfrom collections import OrderedDict\n", "python"
        )
        assert report["checked"] is True
        assert report["unverified_imports"] == []

    def test_python_invented_third_party_is_flagged(self):
        report = check_import_grounding("import os\nimport superfastjson\n", "python")
        assert "superfastjson" in report["unverified_imports"]

    def test_import_grounded_by_rag_context(self):
        rag = "## Reference Examples\nimport requests\nresp = requests.get(url)"
        report = check_import_grounding("import requests", "python", rag_context=rag)
        assert report["unverified_imports"] == []

    def test_import_grounded_by_source_carryover(self):
        report = check_import_grounding(
            "import mycompanylib", "python", source_code="import mycompanylib"
        )
        assert report["unverified_imports"] == []

    def test_java_stdlib_vs_third_party(self):
        report = check_import_grounding(
            "import java.util.List;\nimport org.apache.commons.Foo;", "java"
        )
        assert "org.apache.commons.Foo" in report["unverified_imports"]
        assert "java.util.List" not in report["unverified_imports"]

    def test_go_stdlib_vs_module(self):
        code = 'import (\n  "fmt"\n  "net/http"\n  "github.com/gin-gonic/gin"\n)'
        report = check_import_grounding(code, "go")
        assert report["unverified_imports"] == ["github.com/gin-gonic/gin"]

    def test_go_invented_single_word_package_is_flagged(self):
        # A made-up dot-free package must be flagged now that stdlib detection
        # uses an explicit root list (not "any dot-free import is stdlib").
        code = 'import (\n  "fmt"\n  "fastjson"\n)'
        report = check_import_grounding(code, "go")
        assert "fastjson" in report["unverified_imports"]
        assert "fmt" not in report["unverified_imports"]

    def test_js_relative_and_builtin_are_grounded(self):
        code = (
            "import a from './local.js'\n"
            "import fs from 'fs'\n"
            "import _ from 'lodash'\n"
        )
        report = check_import_grounding(code, "javascript")
        assert report["unverified_imports"] == ["lodash"]

    def test_csharp_system_vs_third_party(self):
        report = check_import_grounding(
            "using System.Text;\nusing Newtonsoft.Json;", "csharp"
        )
        assert "Newtonsoft.Json" in report["unverified_imports"]
        assert "System.Text" not in report["unverified_imports"]

    def test_unsupported_language_reports_not_checked(self):
        report = check_import_grounding("whatever", "cobol")
        assert report["checked"] is False
        assert report["unverified_imports"] == []

    def test_language_alias_is_normalized(self):
        # "py" should resolve to python and still detect invented imports.
        report = check_import_grounding("import notarealpkg", "py")
        assert report["checked"] is True
        assert "notarealpkg" in report["unverified_imports"]


class TestGroundingMatchesWholeNamespaces:
    """Evidence must name the namespace, not merely contain its letters."""

    def test_substring_of_a_longer_identifier_does_not_ground(self):
        # The substring scan this replaces grounded "requests" on any text
        # containing "requestshandler" — a false negative in a check whose whole
        # job is catching invented imports.
        report = check_import_grounding(
            "import requests",
            "python",
            source_code="def requestshandler(): pass",
        )
        assert "requests" in report["unverified_imports"]

    def test_nested_usage_still_grounds_the_package(self):
        report = check_import_grounding(
            "import requests",
            "python",
            rag_context="resp = requests.get(url)",
        )
        assert report["unverified_imports"] == []

    def test_sibling_namespace_is_not_grounded_by_a_stdlib_root(self):
        # "javafx" starts with "java" but is not part of the Java stdlib.
        report = check_import_grounding(
            "import javafx.scene.Node;\nimport java.util.List;", "java"
        )
        assert report["unverified_imports"] == ["javafx.scene.Node"]

    def test_go_module_host_never_resolves_to_a_stdlib_root(self):
        # "image.example.com/x" shares its first letters with stdlib "image",
        # but "." does not close a segment in Go.
        code = 'import (\n  "image/png"\n  "image.example.com/x"\n)'
        report = check_import_grounding(code, "go")
        assert report["unverified_imports"] == ["image.example.com/x"]

    def test_source_carryover_of_a_dotted_namespace(self):
        report = check_import_grounding(
            "import com.mycorp.util.Helper;",
            "java",
            source_code="import com.mycorp.util.Helper;",
        )
        assert report["unverified_imports"] == []

    def test_parent_package_in_context_grounds_a_child_import(self):
        report = check_import_grounding(
            "import com.mycorp.util.Helper;",
            "java",
            rag_context="see com.mycorp.util for helpers",
        )
        assert report["unverified_imports"] == []

    def test_no_imports_reports_checked_with_zero(self):
        report = check_import_grounding("x = 1", "python")
        assert report["checked"] is True
        assert report["total_imports"] == 0
        assert report["unverified_imports"] == []
