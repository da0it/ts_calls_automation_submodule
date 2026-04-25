#!/usr/bin/env python3
import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


SEVERITY_ORDER = {
    "INFO": 0,
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
    "CRITICAL": 4,
}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"report not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_policy(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"policy not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"policy file {path} must be JSON-compatible YAML to avoid external parser dependencies"
        ) from exc


def bandit_issues(path: Path) -> list[dict[str, Any]]:
    data = read_json(path)
    issues = []
    for item in data.get("results", []):
        issues.append(
            {
                "tool": "bandit",
                "severity": str(item.get("issue_severity", "LOW")).upper(),
                "rule": item.get("test_id", "?"),
                "file": item.get("filename", "<unknown>"),
                "line": item.get("line_number", "?"),
                "details": item.get("issue_text", ""),
            }
        )
    return issues


def gosec_issues(path: Path) -> list[dict[str, Any]]:
    data = read_json(path)
    issues = []
    for item in data.get("Issues", []):
        issues.append(
            {
                "tool": "gosec",
                "severity": str(item.get("severity", "LOW")).upper(),
                "rule": item.get("rule_id", "?"),
                "file": item.get("file", "<unknown>"),
                "line": item.get("line", "?"),
                "details": item.get("details", ""),
            }
        )
    return issues


def best_cvss_score(item: dict[str, Any]) -> float | None:
    cvss = item.get("CVSS", {})
    for vendor in ("nvd", "redhat", "ghsa", "vendor"):
        source = cvss.get(vendor, {})
        score = source.get("V3Score")
        if isinstance(score, (int, float)):
            return float(score)
    return None


def trivy_issues(path: Path) -> list[dict[str, Any]]:
    data = read_json(path)
    issues = []
    for result in data.get("Results", []):
        target = result.get("Target", "<unknown>")
        for item in result.get("Vulnerabilities", []) or []:
            issues.append(
                {
                    "tool": "trivy",
                    "severity": str(item.get("Severity", "LOW")).upper(),
                    "rule": item.get("VulnerabilityID", "?"),
                    "file": target,
                    "line": "?",
                    "details": item.get("Title") or item.get("Description") or "",
                    "package": item.get("PkgName", "<unknown>"),
                    "version": item.get("InstalledVersion", "<unknown>"),
                    "fixed_version": item.get("FixedVersion") or "",
                    "cvss": best_cvss_score(item),
                    "status": item.get("Status") or "",
                }
            )
    return issues


def zap_severity(item: dict[str, Any]) -> str:
    risk_code = str(item.get("riskcode", "")).strip()
    if risk_code in {"0", "1", "2", "3"}:
        return {
            "0": "INFO",
            "1": "LOW",
            "2": "MEDIUM",
            "3": "HIGH",
        }[risk_code]

    risk_desc = str(item.get("riskdesc", "")).upper()
    for severity in ("HIGH", "MEDIUM", "LOW", "INFO"):
        if severity in risk_desc:
            return severity
    return "INFO"


def zap_issues(path: Path) -> list[dict[str, Any]]:
    data = read_json(path)
    issues = []
    for site in data.get("site", []) or []:
        site_name = site.get("@name") or site.get("name") or "<unknown>"
        for item in site.get("alerts", []) or []:
            instances = item.get("instances", []) or []
            first_instance = instances[0] if instances else {}
            issues.append(
                {
                    "tool": "zap",
                    "severity": zap_severity(item),
                    "rule": str(item.get("pluginid") or item.get("alertRef") or item.get("alert") or "?"),
                    "file": first_instance.get("uri") or site_name,
                    "line": "?",
                    "details": item.get("alert") or item.get("name") or item.get("desc") or "",
                    "confidence": item.get("confidence") or "",
                    "instances": len(instances),
                    "solution": item.get("solution") or "",
                }
            )
    return issues


def is_blocking(issue: dict[str, Any], threshold: str) -> bool:
    return SEVERITY_ORDER.get(issue["severity"], 0) >= SEVERITY_ORDER[threshold]


def print_summary(name: str, issues: list[dict[str, Any]]) -> None:
    severities = Counter(issue["severity"] for issue in issues)
    summary = (
        f"{name}: total={len(issues)} "
        f"LOW={severities.get('LOW', 0)} "
        f"MEDIUM={severities.get('MEDIUM', 0)} "
        f"HIGH={severities.get('HIGH', 0)} "
        f"CRITICAL={severities.get('CRITICAL', 0)}"
    )
    if severities.get("INFO", 0):
        summary += f" INFO={severities.get('INFO', 0)}"
    print(summary)


def issue_matches_exception(issue: dict[str, Any], exception: dict[str, Any]) -> bool:
    vulnerability_id = exception.get("vulnerability_id")
    package = exception.get("package")
    return issue.get("rule") == vulnerability_id and (
        package is None or issue.get("package") == package
    )


def filter_ignored(
    issues: list[dict[str, Any]], policy: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    exceptions = policy.get("policy", {}).get("ignored_findings", [])
    ignored: list[dict[str, Any]] = []
    effective: list[dict[str, Any]] = []
    for issue in issues:
        matched = next((item for item in exceptions if issue_matches_exception(issue, item)), None)
        if matched:
            ignored.append({**issue, "ignore_reason": matched.get("reason", "documented exception")})
            continue
        effective.append(issue)
    return ignored, effective


def is_banned_component(issue: dict[str, Any], policy: dict[str, Any]) -> bool:
    banned = policy.get("policy", {}).get("banned_components", [])
    package = issue.get("package")
    version = issue.get("version")
    if not package or not version:
        return False
    return any(item.get("name") == package and item.get("version") == version for item in banned)


def policy_warns(issue: dict[str, Any], policy: dict[str, Any]) -> bool:
    policy_data = policy.get("policy", {})
    warn_severities = {item.upper() for item in policy_data.get("warn_severities", [])}
    warn_cvss_score = policy_data.get("warn_cvss_score")
    severity = issue.get("severity", "LOW")
    if severity in warn_severities:
        return True
    cvss = issue.get("cvss")
    return isinstance(cvss, (int, float)) and warn_cvss_score is not None and cvss >= warn_cvss_score


def policy_blocks(issue: dict[str, Any], policy: dict[str, Any]) -> bool:
    policy_data = policy.get("policy", {})
    auto_block = {item.upper() for item in policy_data.get("auto_block_severities", [])}
    max_cvss_score = policy_data.get("max_cvss_score")
    severity = issue.get("severity", "LOW")
    if severity in auto_block:
        return True
    if is_banned_component(issue, policy):
        return True
    cvss = issue.get("cvss")
    return isinstance(cvss, (int, float)) and max_cvss_score is not None and cvss >= max_cvss_score


def format_issue(issue: dict[str, Any]) -> str:
    package = issue.get("package")
    version = issue.get("version")
    fixed_version = issue.get("fixed_version")
    cvss = issue.get("cvss")
    confidence = issue.get("confidence")
    instances = issue.get("instances")
    extras = []
    if package and version:
        extras.append(f"{package}@{version}")
    if fixed_version:
        extras.append(f"fixed={fixed_version}")
    if isinstance(cvss, (int, float)):
        extras.append(f"cvss={cvss}")
    if confidence:
        extras.append(f"confidence={confidence}")
    if isinstance(instances, int) and instances > 0:
        extras.append(f"instances={instances}")
    suffix = f" [{' '.join(extras)}]" if extras else ""
    return (
        f"{issue['tool']} {issue['severity']} {issue['rule']} "
        f"{issue['file']}:{issue['line']} {issue['details']}{suffix}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Security Gate for Bandit, gosec, Trivy and ZAP reports")
    parser.add_argument("--bandit", action="append", default=[], type=Path, help="Path to Bandit JSON report")
    parser.add_argument("--gosec", action="append", default=[], type=Path, help="Path to gosec JSON report")
    parser.add_argument("--trivy", action="append", default=[], type=Path, help="Path to Trivy JSON report")
    parser.add_argument("--zap", action="append", default=[], type=Path, help="Path to OWASP ZAP JSON report")
    parser.add_argument("--policy", type=Path, default=None, help="Path to JSON-compatible YAML security policy")
    parser.add_argument(
        "--threshold",
        choices=("low", "medium", "high", "critical"),
        default="high",
        help="Minimal severity that blocks the pipeline when no policy is provided",
    )
    args = parser.parse_args()

    threshold = args.threshold.upper()
    policy = load_policy(args.policy)
    all_issues: list[dict[str, Any]] = []

    for path in args.bandit:
        issues = bandit_issues(path)
        print_summary(path.name, issues)
        all_issues.extend(issues)

    for path in args.gosec:
        issues = gosec_issues(path)
        print_summary(path.name, issues)
        all_issues.extend(issues)

    for path in args.trivy:
        issues = trivy_issues(path)
        print_summary(path.name, issues)
        all_issues.extend(issues)

    for path in args.zap:
        issues = zap_issues(path)
        print_summary(path.name, issues)
        all_issues.extend(issues)

    ignored, effective_issues = filter_ignored(all_issues, policy)
    warnings = [issue for issue in effective_issues if policy_warns(issue, policy)] if policy else []
    if policy:
        blockers = [issue for issue in effective_issues if policy_blocks(issue, policy)]
    else:
        blockers = [issue for issue in effective_issues if is_blocking(issue, threshold)]

    print(f"Security Gate threshold: {threshold}")
    if policy:
        print(f"Security policy loaded: {args.policy.name}")
    print(f"All issues: {len(all_issues)}")
    print(f"Ignored issues: {len(ignored)}")
    print(f"Effective issues: {len(effective_issues)}")
    if policy:
        print(f"Warnings: {len(warnings)}")
    print(f"Blocking issues: {len(blockers)}")

    for issue in ignored:
        print(f"IGNORED {format_issue(issue)} reason={issue['ignore_reason']}")

    for issue in blockers:
        print(format_issue(issue))

    return 1 if blockers else 0


if __name__ == "__main__":
    sys.exit(main())
