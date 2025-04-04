import React, { useEffect, useState } from "react";
import "./RiskTable.css";

type RiskEntry = {
  company: string;
  year: number;
  riskScore: number | null;
  riskDetails: { topic: string; risk: string; mitigation: string }[];
};

type SortKey = keyof Pick<RiskEntry, "company" | "year" | "riskScore">;

const RiskTable: React.FC = () => {
  const [data, setData] = useState<RiskEntry[]>([]);
  const [currentPage, setCurrentPage] = useState(1);
  const [sortKey, setSortKey] = useState<SortKey>("year");
  const [sortAsc, setSortAsc] = useState(true);
  const [filterCompany, setFilterCompany] = useState("");
  const [filterYear, setFilterYear] = useState("");
  const rowsPerPage = 10;

  useEffect(() => {
    const loadRiskData = async () => {
      const analysisRes = await fetch("/results/risk_analysis.json");
      const scoreRes = await fetch("/results/aggregated_risk_scores.json");

      const analysisJson = await analysisRes.json();
      const scoresJson = await scoreRes.json();

      const flatData: RiskEntry[] = [];

      for (const company of Object.keys(analysisJson)) {
        const years = Object.keys(analysisJson[company]);

        for (const year of years) {
          const yearData = analysisJson[company][year];
          const topics = Object.keys(yearData || {});

          const riskDetails = topics.map((topic) => ({
            topic,
            risk: yearData[topic]?.potential_transition_risks || "N/A",
            mitigation: yearData[topic]?.mitigation_strategies || "N/A",
          }));

          flatData.push({
            company,
            year: parseInt(year),
            riskScore: scoresJson[company]?.[year] ?? null,
            riskDetails,
          });
        }
      }

      setData(flatData);
    };

    loadRiskData();
  }, []);

  // Filtered
  const filteredData = data.filter((row) => {
    return (
      (filterCompany === "" || row.company.toLowerCase().includes(filterCompany.toLowerCase())) &&
      (filterYear === "" || row.year.toString().includes(filterYear))
    );
  });

  useEffect(() => {
    setCurrentPage(1);
  }, [filterCompany, filterYear]);

  // Sorted
  const sortedData = [...filteredData].sort((a, b) => {
    const valA = a[sortKey];
    const valB = b[sortKey];
    if (valA === null) return 1;
    if (valB === null) return -1;
    if (valA === valB) return 0;
    return sortAsc ? (valA > valB ? 1 : -1) : (valA < valB ? 1 : -1);
  });

  // Paginated
  const totalPages = Math.ceil(sortedData.length / rowsPerPage);
  const paginatedData = sortedData.slice(
    (currentPage - 1) * rowsPerPage,
    currentPage * rowsPerPage
  );

  const toggleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortAsc(!sortAsc);
    } else {
      setSortKey(key);
      setSortAsc(true);
    }
  };

  const handlePrev = () => setCurrentPage((prev) => Math.max(prev - 1, 1));
  const handleNext = () => setCurrentPage((prev) => Math.min(prev + 1, totalPages));

  return (
    <div className="risk-table-container">
      <h2>Risk Analysis Table</h2>

      <div className="score-table-filters">
        <input
          type="text"
          placeholder="Filter by company"
          value={filterCompany}
          onChange={(e) => setFilterCompany(e.target.value)}
        />
        <input
          type="text"
          placeholder="Filter by year"
          value={filterYear}
          onChange={(e) => setFilterYear(e.target.value)}
        />
      </div>

      <div className="pagination">
        <button onClick={handlePrev} disabled={currentPage === 1}>
          Previous
        </button>
        <span>
          Page {currentPage} of {totalPages}
        </span>
        <button onClick={handleNext} disabled={currentPage === totalPages}>
          Next
        </button>
      </div>

      <table className="risk-table">
        <thead>
          <tr>
            <th className="company sortable" onClick={() => toggleSort("company")}>
              Company {sortKey === "company" && (sortAsc ? "↑" : "↓")}
            </th>
            <th className="year sortable" onClick={() => toggleSort("year")}>
              Year {sortKey === "year" && (sortAsc ? "↑" : "↓")}
            </th>
            <th className="score sortable" onClick={() => toggleSort("riskScore")}>
              Risk Score {sortKey === "riskScore" && (sortAsc ? "↑" : "↓")}
            </th>
            <th className="risk">Risk</th>
            <th className="mitigation">Mitigation</th>
          </tr>
        </thead>
        <tbody>
          {paginatedData.map((entry, idx) => (
            <tr key={idx}>
              <td>{entry.company}</td>
              <td>{entry.year}</td>
              <td>{entry.riskScore ?? "N/A"}</td>
              <td>
                {entry.riskDetails.length > 0 ? (
                  entry.riskDetails.map((detail, i) => (
                    <div key={i} style={{ marginBottom: "0.5rem" }}>
                      <strong>{detail.topic}</strong>
                      <div>Risk: {detail.risk}</div>
                    </div>
                  ))
                ) : (
                  <span>No risks listed</span>
                )}
              </td>
              <td>
                {entry.riskDetails.length > 0 ? (
                  entry.riskDetails.map((detail, i) => (
                    <div key={i} style={{ marginBottom: "0.5rem" }}>
                      <strong>{detail.topic}</strong>
                      <div>Mitigation: {detail.mitigation}</div>
                    </div>
                  ))
                ) : (
                  <span>No risks listed</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default RiskTable;
