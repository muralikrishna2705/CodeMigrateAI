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
