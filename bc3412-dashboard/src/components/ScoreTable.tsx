import React, { useEffect, useState } from "react";
import "./ScoreTable.css";

type RankingRecord = {
  company: string;
  year: number;
  score: number | null;
  rank: number | null;
  mitigation: string;
};

type SortKey = keyof Pick<RankingRecord, "company" | "year" | "score" | "rank">;

const ScoreTable: React.FC = () => {
  const [data, setData] = useState<RankingRecord[]>([]);
  const [filterCompany, setFilterCompany] = useState("");
  const [filterYear, setFilterYear] = useState("");
  const [currentPage, setCurrentPage] = useState(1);
  const [sortKey, setSortKey] = useState<SortKey>("year");
  const [sortAsc, setSortAsc] = useState(true);
  const rowsPerPage = 10;

  useEffect(() => {
    const loadData = async () => {
      const rankingRes = await fetch("results/rankings.json");
      const rankingsJson = await rankingRes.json();

      const companies = Object.values(rankingsJson[Object.keys(rankingsJson)[0]]).map((_, i) =>
        Object.keys(rankingsJson[Object.keys(rankingsJson)[0]])[i]
      );

      const mitigationRules: Record<string, Record<string, string>> = {};
      for (const company of companies) {
        const rulesRes = await fetch(`/results/${company}/${company}_phased_scenario_rules.json`);
        if (rulesRes.ok) {
          const rulesJson = await rulesRes.json();
          mitigationRules[company] = rulesJson;
        }
      }

      const flatData: RankingRecord[] = [];

      for (const year of Object.keys(rankingsJson)) {
        for (const company of Object.keys(rankingsJson[year])) {
          const entry = rankingsJson[year][company];
          const mitigationEntries = mitigationRules[company]?.[year];
          const mitigation = mitigationEntries
            ? Object.entries(mitigationEntries)
                .map(([key, val]) => `${key}: ${val}`)
                .join("; ")
            : "";
          flatData.push({
            company,
            year: parseInt(year),
            score: entry.phased_score,
            rank: entry.phased_rank,
            mitigation,
          });
        }
      }

      setData(flatData);
    };

    loadData();
  }, []);

  // Filtering
  const filteredData = data.filter((row) => {
    return (
      (filterCompany === "" || row.company.includes(filterCompany)) &&
      (filterYear === "" || row.year.toString().includes(filterYear))
    );
  });

  // Sorting
  const sortedData = [...filteredData].sort((a, b) => {
    const valA = a[sortKey];
    const valB = b[sortKey];
    if (valA === null) return 1;
    if (valB === null) return -1;
    if (valA === valB) return 0;
    return sortAsc ? (valA > valB ? 1 : -1) : (valA < valB ? 1 : -1);
  });

  // Pagination
  const totalPages = Math.ceil(sortedData.length / rowsPerPage);
  const paginatedData = sortedData.slice(
    (currentPage - 1) * rowsPerPage,
    currentPage * rowsPerPage
  );

  const handlePrev = () => setCurrentPage((prev) => Math.max(prev - 1, 1));
  const handleNext = () => setCurrentPage((prev) => Math.min(prev + 1, totalPages));

  useEffect(() => {
    setCurrentPage(1);
  }, [filterCompany, filterYear]);

  const toggleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortAsc(!sortAsc);
    } else {
      setSortKey(key);
      setSortAsc(true);
    }
  };

  return (
    <div className="score-table-container">
      <h2>Important Metrics, Score Analysis & Ranking Table</h2>
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
      <table className="score-table">
        <thead>
          <tr>
            <th className="company sortable" onClick={() => toggleSort("company")}>
              Company {sortKey === "company" && (sortAsc ? "↑" : "↓")}
            </th>
            <th className="year sortable" onClick={() => toggleSort("year")}>
              Year {sortKey === "year" && (sortAsc ? "↑" : "↓")}
            </th>
            <th className="score sortable" onClick={() => toggleSort("score")}>
              Score {sortKey === "score" && (sortAsc ? "↑" : "↓")}
            </th>
            <th className="rank sortable" onClick={() => toggleSort("rank")}>
              Rank {sortKey === "rank" && (sortAsc ? "↑" : "↓")}
            </th>
            <th className="mitigation">Important Metrics To Take Note</th>
          </tr>
        </thead>
        <tbody>
          {paginatedData.map((row, idx) => (
            <tr key={idx}>
              <td>{row.company}</td>
              <td>{row.year}</td>
              <td>{row.score ?? "N/A"}</td>
              <td>{row.rank ?? "N/A"}</td>
              <td>
                {row.mitigation
                    ? row.mitigation.split(";").map((item, idx) => (
                        <div key={idx}>{item.trim()}</div>
                    ))
                    : "No action listed"}
                </td>


            </tr>
          ))}
        </tbody>
      </table>

      
    </div>
  );
};

export default ScoreTable;
