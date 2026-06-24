import { useState, useEffect, useCallback, useRef } from "react";
import Header    from "./components/Header.jsx";
import Sidebar   from "./components/Sidebar.jsx";
import Editor    from "./components/Editor.jsx";
import Output    from "./components/Output.jsx";
import AgentLog  from "./components/AgentLog.jsx";
import { useMigrationStream } from "./hooks/useMigrationStream.js";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

// Sample code snippets pre-loaded for the evaluator / demo
export const EXAMPLES = {
  "Java 7 → Java 17": {
    code: `import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.List;
import java.util.HashMap;
import java.util.Map;

public class EmployeeService {

    private List<String> employees = new ArrayList<String>();
    private Map<String, Integer> salaries = new HashMap<String, Integer>();

    public void addEmployee(String name, int salary) {
        if (name != null && !name.isEmpty()) {
            employees.add(name);
            salaries.put(name, salary);
        }
    }

    public List<String> getSortedEmployees() {
        List<String> sorted = new ArrayList<String>(employees);
        Collections.sort(sorted);
        return sorted;
    }

    public String findEmployee(String prefix) {
        for (String emp : employees) {
            if (emp.startsWith(prefix)) {
                return emp;
            }
        }
        return null;
    }

    public Date getReportDate() {
        return new Date();
    }

    public void printRoster() {
        for (String emp : employees) {
            System.out.println("Employee: " + emp + " | Salary: " + salaries.get(emp));
        }
    }
}`,
    sl: "java", sv: "7", tl: "java", tv: "17",
  },

  "Java → Python": {
    code: `import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

public class InventoryManager {

    private Map<String, Integer> stock = new HashMap<>();
    private Map<String, Double>  prices = new HashMap<>();

    public void addItem(String item, int qty, double price) {
        stock.put(item, stock.getOrDefault(item, 0) + qty);
        prices.put(item, price);
    }

    public Optional<Integer> getStock(String item) {
        return Optional.ofNullable(stock.get(item));
    }

    public double getTotalValue() {
        double total = 0.0;
        for (Map.Entry<String, Integer> entry : stock.entrySet()) {
            total += entry.getValue() * prices.getOrDefault(entry.getKey(), 0.0);
        }
        return total;
    }

    public void printInventory() {
        for (String item : stock.keySet()) {
            System.out.printf("%-20s qty=%-5d price=%.2f%n",
                item, stock.get(item), prices.get(item));
        }
    }
}`,
    sl: "java", sv: "8", tl: "python", tv: "3.12",
  },

  "Python 2.7 → Python 3.12": {
    code: `# Python 2.7 legacy script
import urllib2
import json

class DataFetcher(object):

    def __init__(self, base_url):
        self.base_url = base_url
        self.cache    = {}

    def fetch(self, endpoint):
        if endpoint in self.cache:
            print "Cache hit: %s" % endpoint
            return self.cache[endpoint]

        url  = self.base_url + endpoint
        resp = urllib2.urlopen(url)
        data = json.loads(resp.read())
        self.cache[endpoint] = data
        print "Fetched: %s" % endpoint
        return data

    def clear_cache(self):
        self.cache = {}
        print "Cache cleared."

    def items_count(self):
        return self.cache.keys().__len__()`,
    sl: "python", sv: "2.7", tl: "python", tv: "3.12",
  },
};

export default function App() {
  // ── State ──────────────────────────────────────────────────────────────────
  const [sourceCode, setSourceCode]   = useState(EXAMPLES["Java 7 → Java 17"].code);
  const [srcLang, setSrcLang]         = useState("java");
  const [srcVer,  setSrcVer]          = useState("7");
  const [tgtLang, setTgtLang]         = useState("java");
  const [tgtVer,  setTgtVer]          = useState("17");

  const [ollamaStatus, setOllamaStatus] = useState("checking");  // checking | ok | down
  const [activePanel, setActivePanel] = useState("editor");      // editor | output

  const healthTimer = useRef(null);

  // ── Streaming migration hook ───────────────────────────────────────────────
  const {
    result,
    streamBuffer,
    agentProgress,
    loading,
    error,
    runMigration,
    cancel,
  } = useMigrationStream(API_BASE);

  // ── Health polling ─────────────────────────────────────────────────────────
  const checkHealth = useCallback(async () => {
    try {
      const r = await fetch(`${API_BASE}/health`);
      const d = await r.json();
      setOllamaStatus(d.ollama === "connected" ? "ok" : "down");
    } catch {
      setOllamaStatus("down");
    }
  }, []);

  useEffect(() => {
    checkHealth();
    healthTimer.current = setInterval(checkHealth, 15_000);
    return () => clearInterval(healthTimer.current);
  }, [checkHealth]);

  // ── Load example ───────────────────────────────────────────────────────────
  const loadExample = useCallback((key) => {
    const ex = EXAMPLES[key];
    setSourceCode(ex.code);
    setSrcLang(ex.sl); setSrcVer(ex.sv);
    setTgtLang(ex.tl); setTgtVer(ex.tv);
    setActivePanel("editor");
  }, []);

  // ── Run migration wrapper ──────────────────────────────────────────────────
  const handleRunMigration = useCallback(() => {
    if (!sourceCode.trim()) {
      // The hook will handle error state
      return;
    }
    setActivePanel("output");
    runMigration({
      source_code: sourceCode,
      source_language: srcLang,
      source_version: srcVer,
      target_language: tgtLang,
      target_version: tgtVer,
    });
  }, [sourceCode, srcLang, srcVer, tgtLang, tgtVer, runMigration]);

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div style={styles.shell}>
      <Header
        ollamaStatus={ollamaStatus}
        examples={Object.keys(EXAMPLES)}
        onLoadExample={loadExample}
      />

      <div style={styles.body}>
        {/* ── LEFT SIDEBAR ── */}
        <Sidebar
          srcLang={srcLang} setSrcLang={setSrcLang}
          srcVer={srcVer}   setSrcVer={setSrcVer}
          tgtLang={tgtLang} setTgtLang={setTgtLang}
          tgtVer={tgtVer}   setTgtVer={setTgtVer}
          loading={loading}
          ollamaStatus={ollamaStatus}
          apiError={error}
          result={result}
          agentProgress={agentProgress}
          onRun={handleRunMigration}
        />

        {/* ── MAIN PANELS ── */}
        <div style={styles.panels}>
          {/* Tab bar */}
          <div style={styles.tabBar}>
            <button
              style={{...styles.tab, ...(activePanel === "editor" ? styles.tabActive : {})}}
              onClick={() => setActivePanel("editor")}
            >
              Source Code
            </button>
            <button
              style={{
                ...styles.tab,
                ...(activePanel === "output" ? styles.tabActive : {}),
                ...(result ? styles.tabHasResult : {}),
                opacity: (!result && !loading) ? 0.35 : 1,
              }}
              onClick={() => setActivePanel("output")}
              disabled={!result && !loading}
            >
              Migrated Code {result?.success && "✓"}
            </button>

            {/* Agent log inline tab */}
            {(result || agentProgress.length > 0) && (
              <div style={styles.agentTabArea}>
                <AgentLog reports={result?.reports || []} compact />
              </div>
            )}
          </div>

          {/* Panel content */}
          <div style={styles.panelBody}>
            {activePanel === "editor" ? (
              <Editor
                code={sourceCode}
                onChange={setSourceCode}
                language={srcLang}
                version={srcVer}
              />
            ) : (
              <Output
                result={result}
                streamBuffer={streamBuffer}
                loading={loading}
                language={tgtLang}
                version={tgtVer}
                agentProgress={agentProgress}
              />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Inline layout styles (no external CSS framework dependency) ────────────────
const styles = {
  shell: {
    display: "flex", flexDirection: "column",
    height: "100vh", overflow: "hidden",
    background: "var(--bg-root)",
  },
  body: {
    display: "flex", flex: 1, overflow: "hidden",
  },
  panels: {
    display: "flex", flexDirection: "column",
    flex: 1, overflow: "hidden",
  },
  tabBar: {
    display: "flex", alignItems: "center",
    background: "var(--bg-base)",
    borderBottom: "1px solid var(--border)",
    padding: "0 16px",
    gap: 4,
    minHeight: 44,
    flexShrink: 0,
  },
  tab: {
    padding: "0 18px", height: 44,
    background: "none", border: "none",
    borderBottom: "2px solid transparent",
    color: "var(--text-muted)", fontSize: 13,
    fontFamily: "var(--font-ui)", fontWeight: 500,
    cursor: "pointer", transition: "all .15s",
    marginBottom: -1,
  },
  tabActive: {
    color: "var(--accent)",
    borderBottomColor: "var(--accent)",
  },
  tabHasResult: {
    color: "var(--text-normal)",
  },
  agentTabArea: {
    marginLeft: "auto",
  },
  panelBody: {
    flex: 1, display: "flex", overflow: "hidden",
  },
};
