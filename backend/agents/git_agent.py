"""
agents/git_agent.py
--------------------
Git Versiyon Kontrol Ajanı (Git Commit Agent)

Sorumluluk (SRP):
  Yalnızca Git versiyon kontrolü ile ilgilenir. Projedeki değişiklikleri
  analiz eder, 'Conventional Commits' standardına uygun atomik commit'ler atar.

Fail-Fast Politikası:
  - Commitlenecek değişiklik yoksa → logla ve dur.
  - Git komutu başarısız olursa → GitCommandError fırlat, logla ve dur.
  - İzin verilmeyen shell komutu → SecurityError fırlat.

SOLID Prensipleri:
  - SRP : Tek sorumlu = versiyon kontrolü.
  - OCP : Yeni commit tipi sınıflandırması _RULES listesine satır eklenerek yapılır.
  - LSP : CommitGroup bir dataclass; alt sınıflarla yer değiştirilebilir.
  - ISP : GitShellTool yalnızca run() metodunu sunar; fazlası yok.
  - DIP : GitCommitAgent somut subprocess'e değil, GitShellTool soyutlamasına bağımlıdır.

Kullanım (izole mod):
    from agents.git_agent import GitCommitAgent
    agent = GitCommitAgent(repo_path="/path/to/repo")
    summary = agent.auto_commit()
    print(summary)
"""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Özel İstisnalar
# ---------------------------------------------------------------------------

class GitAgentError(RuntimeError):
    """Git Ajanı temel hata sınıfı."""


class GitCommandError(GitAgentError):
    """Git komutu sıfırdan farklı çıkış kodu döndürdü."""

    def __init__(self, command: str, returncode: int, stderr: str) -> None:
        self.command = command
        self.returncode = returncode
        self.stderr = stderr
        super().__init__(
            f"Git komutu başarısız (kod {returncode}): `{command}`\n{stderr}"
        )


class SecurityError(GitAgentError):
    """İzin listesi dışı bir komut çalıştırılmaya çalışıldı."""


class NothingToCommitError(GitAgentError):
    """Commitlenecek değişiklik bulunamadı — fail-fast."""


# ---------------------------------------------------------------------------
# Conventional Commit Tipleri
# ---------------------------------------------------------------------------

class CommitType(str, Enum):
    """
    Conventional Commits v1.0 – izin verilen tipler.
    https://www.conventionalcommits.org/en/v1.0.0/
    """
    FEAT     = "feat"       # yeni özellik
    FIX      = "fix"        # hata düzeltmesi
    REFACTOR = "refactor"   # davranış değişikliği olmadan yeniden yapılandırma
    CHORE    = "chore"      # yapılandırma / bağımlılık / build
    DOCS     = "docs"       # dokümantasyon
    TEST     = "test"       # test ekleme / düzenleme
    STYLE    = "style"      # biçimlendirme (mantık yok)
    PERF     = "perf"       # performans iyileştirmesi
    CI       = "ci"         # CI/CD pipeline
    REVERT   = "revert"     # geri alma


# ---------------------------------------------------------------------------
# Sınıflandırma Kuralları (OCP — satır ekleyerek genişletilebilir)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _ClassificationRule:
    """
    Bir dosya yolunun CommitType'ını belirleyen kural.
    Tüm alanlar OR mantığıyla değil AND mantığıyla eşleştirilir.
    """
    path_pattern: str       # re.search ile dosya yolu üzerinde çalışır
    git_status: str | None  # 'A'=added, 'M'=modified, 'D'=deleted, None=hepsi
    commit_type: CommitType
    scope_hint: str         # commit mesajında kullanılacak kapsam önerisi
    description_template: str


# Öncelik sırası önemlidir: daha özgül kurallar üstte olmalı
_RULES: list[_ClassificationRule] = [
    # Test dosyaları
    _ClassificationRule(r"tests?/|test_.*\.py$|_test\.py$",  None,  CommitType.TEST,     "tests",        "add/update test coverage"),
    # CI/CD
    _ClassificationRule(r"\.github/|\.gitlab-ci|Dockerfile|docker-compose", None, CommitType.CI, "ci",  "update CI/CD configuration"),
    # Dokümantasyon
    _ClassificationRule(r"README|CHANGELOG|\.md$|docs/",      None,  CommitType.DOCS,     "docs",         "update documentation"),
    # Bağımlılıklar / yapılandırma
    _ClassificationRule(r"requirements.*\.txt$|setup\.(py|cfg)|pyproject\.toml|\.env", None, CommitType.CHORE, "deps", "update dependencies/config"),
    # Paket init'leri — yeni eklenirse feat, değiştirilirse chore
    _ClassificationRule(r"__init__\.py$",                      "A",   CommitType.FEAT,     "package",      "expose new public API"),
    _ClassificationRule(r"__init__\.py$",                      "M",   CommitType.CHORE,    "package",      "update package exports"),
    # Agent katmanı
    _ClassificationRule(r"agents/.*\.py$",                     "A",   CommitType.FEAT,     "agents",       "add new agent module"),
    _ClassificationRule(r"agents/.*\.py$",                     "M",   CommitType.REFACTOR, "agents",       "refactor agent logic"),
    # Servis katmanı
    _ClassificationRule(r"services/.*\.py$",                   "A",   CommitType.FEAT,     "services",     "add new service"),
    _ClassificationRule(r"services/.*\.py$",                   "M",   CommitType.REFACTOR, "services",     "refactor service layer"),
    # API / route katmanı
    _ClassificationRule(r"api/.*\.py$",                        "A",   CommitType.FEAT,     "api",          "add new endpoint"),
    _ClassificationRule(r"api/.*\.py$",                        "M",   CommitType.FIX,      "api",          "fix api handler"),
    # ML motoru
    _ClassificationRule(r"ml_engine/.*\.py$",                  "A",   CommitType.FEAT,     "ml",           "add ml component"),
    _ClassificationRule(r"ml_engine/.*\.py$",                  "M",   CommitType.REFACTOR, "ml",           "refactor ml pipeline"),
    # Repository / DB katmanı
    _ClassificationRule(r"repositories/.*\.py$",               "A",   CommitType.FEAT,     "repository",   "add repository"),
    _ClassificationRule(r"repositories/.*\.py$",               "M",   CommitType.REFACTOR, "repository",   "refactor data access"),
    # Şemalar / modeller
    _ClassificationRule(r"schemas/.*\.py$",                    None,  CommitType.REFACTOR, "schemas",      "update data schemas"),
    _ClassificationRule(r"models/.*\.py$",                     None,  CommitType.REFACTOR, "models",       "update domain models"),
    # main.py
    _ClassificationRule(r"main\.py$",                          "A",   CommitType.FEAT,     "app",          "bootstrap FastAPI application"),
    _ClassificationRule(r"main\.py$",                          "M",   CommitType.CHORE,    "app",          "update app configuration"),
    # Silinen her şey
    _ClassificationRule(r".*",                                 "D",   CommitType.CHORE,    "cleanup",      "remove obsolete file"),
    # Geri kalan değiştirilen dosyalar
    _ClassificationRule(r".*",                                 "M",   CommitType.FIX,      "misc",         "apply corrections"),
    # Geri kalan eklenen dosyalar
    _ClassificationRule(r".*",                                 "A",   CommitType.FEAT,     "misc",         "add new resource"),
]


# ---------------------------------------------------------------------------
# Veri Transferi Nesneleri
# ---------------------------------------------------------------------------

@dataclass
class ChangeRecord:
    """git status --porcelain'den gelen tek dosya değişikliği."""
    status: str   # 'A', 'M', 'D', 'R', '?' ...
    path: str     # repo'ya göreli yol


@dataclass
class CommitGroup:
    """Atomik bir commit'i oluşturan dosya grubu."""
    commit_type: CommitType
    scope: str
    description: str
    files: list[str] = field(default_factory=list)
    breaking_change: bool = False

    @property
    def message(self) -> str:
        """
        Conventional Commits formatında mesaj üretir.
        Örnek: feat(agents): add new agent module
        """
        bang = "!" if self.breaking_change else ""
        return f"{self.commit_type.value}{bang}({self.scope}): {self.description}"


@dataclass
class CommitResult:
    """Tek bir git commit işleminin sonucu."""
    success: bool
    message: str
    files: list[str]
    sha: str = ""
    error: str = ""

    def __str__(self) -> str:
        status = "✓" if self.success else "✗"
        return f"  [{status}] {self.message}" + (f"\n      ERROR: {self.error}" if self.error else "")


@dataclass
class AutoCommitSummary:
    """auto_commit() metodunun nihai özeti."""
    started_at: datetime
    finished_at: datetime
    repo_path: str
    total_changes: int
    groups_found: int
    commits_made: int
    commits_failed: int
    results: list[CommitResult] = field(default_factory=list)

    def __str__(self) -> str:
        duration = (self.finished_at - self.started_at).total_seconds()
        lines = [
            "═" * 60,
            f"  Git Commit Agent — Auto Commit Özeti",
            f"  Repo     : {self.repo_path}",
            f"  Süre     : {duration:.2f}s",
            f"  Değişiklik: {self.total_changes} dosya → {self.groups_found} grup",
            f"  Başarılı : {self.commits_made} commit",
            f"  Başarısız: {self.commits_failed} commit",
            "─" * 60,
        ]
        for r in self.results:
            lines.append(str(r))
        lines.append("═" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Güvenli Shell Aracı  (Langchain ShellTool muadili — DIP)
# ---------------------------------------------------------------------------

class GitShellTool:
    """
    Yalnızca beyaz listedeki git komutlarını çalıştıran güvenli subprocess sarmalayıcısı.
    Langchain ShellTool'un proje-içi, bağımlılıksız eşdeğeri.

    Güvenlik modeli:
      - İzin listesi dışı herhangi bir komut → SecurityError (fail-fast).
      - Tüm kullanıcı girdileri argüman listesiyle iletilir (injection yok).
      - Shell=False — sistem kabuğu çağrılmaz.
    """

    # İzin verilen git alt-komutları (whitelist)
    _ALLOWED_SUBCOMMANDS: frozenset[str] = frozenset({
        "status", "add", "commit", "diff", "log",
        "rev-parse", "config", "ls-files",
    })

    def __init__(self, repo_path: str | Path, timeout: int = 30) -> None:
        """
        Args:
            repo_path: Git deposunun kök dizini.
            timeout  : Komut zaman aşımı (saniye).
        """
        self._repo_path = Path(repo_path).resolve()
        self._timeout = timeout

    def run(self, *args: str) -> str:
        """
        Güvenli bir `git <args>` komutu çalıştırır.

        Args:
            *args: git'e iletilecek argümanlar (ör. "status", "--porcelain").

        Returns:
            stdout çıktısı (strip edilmiş).

        Raises:
            SecurityError    : İzin verilmeyen bir komut denenirse.
            GitCommandError  : Git sıfırdan farklı çıkış kodu döndürürse.
        """
        if not args:
            raise SecurityError("Boş komut iletildi.")

        subcommand = args[0]
        if subcommand not in self._ALLOWED_SUBCOMMANDS:
            raise SecurityError(
                f"İzin verilmeyen git alt-komutu: '{subcommand}'. "
                f"İzin verilenler: {sorted(self._ALLOWED_SUBCOMMANDS)}"
            )

        full_cmd = ["git"] + list(args)
        logger.debug("ShellTool çalıştırıyor: %s", " ".join(full_cmd))

        try:
            result = subprocess.run(
                full_cmd,
                cwd=str(self._repo_path),
                capture_output=True,
                text=True,
                timeout=self._timeout,
                check=False,            # Hata kontrolü manuel yapılır
                shell=False,            # Injection koruması
            )
        except subprocess.TimeoutExpired:
            raise GitCommandError(
                " ".join(full_cmd), -1,
                f"Komut {self._timeout}s içinde tamamlanamadı."
            )
        except FileNotFoundError:
            raise GitCommandError(
                " ".join(full_cmd), -1,
                "git çalıştırılabilir dosyası bulunamadı. Git kurulu mu?"
            )

        if result.returncode != 0:
            raise GitCommandError(
                " ".join(full_cmd), result.returncode, result.stderr.strip()
            )

        return result.stdout.strip()


# ---------------------------------------------------------------------------
# Conventional Commit Sınıflandırıcısı
# ---------------------------------------------------------------------------

class ConventionalCommitClassifier:
    """
    Değişiklik kayıtlarını CommitGroup listesine dönüştürür.

    Her CommitGroup bir atomik git commit'e karşılık gelir.
    Gruplama mantığı: (commit_type, scope) çiftine göre.
    """

    def classify(self, changes: list[ChangeRecord]) -> list[CommitGroup]:
        """
        Args:
            changes: git status'tan gelen ChangeRecord listesi.

        Returns:
            list[CommitGroup]: Atomik commit grupları, (type, scope) sıralı.
        """
        groups: dict[tuple[CommitType, str], CommitGroup] = {}

        for change in changes:
            rule = self._match_rule(change)
            key = (rule.commit_type, rule.scope_hint)

            if key not in groups:
                groups[key] = CommitGroup(
                    commit_type=rule.commit_type,
                    scope=rule.scope_hint,
                    description=rule.description_template,
                )
            groups[key].files.append(change.path)

        result = list(groups.values())
        logger.info(
            "Sınıflandırma tamamlandı: %d değişiklik → %d grup",
            len(changes), len(result),
        )
        return result

    def _match_rule(self, change: ChangeRecord) -> _ClassificationRule:
        """İlk eşleşen kuralı döner; hiçbiri eşleşmezse genel kural kullanılır."""
        for rule in _RULES:
            status_matches = (
                rule.git_status is None or
                rule.git_status == change.status
            )
            path_matches = bool(re.search(rule.path_pattern, change.path, re.IGNORECASE))
            if status_matches and path_matches:
                return rule

        # Fallback — hiçbir kural eşleşmedi (teorik olarak olmamalı)
        return _ClassificationRule(r".*", None, CommitType.CHORE, "misc", "miscellaneous change")


# ---------------------------------------------------------------------------
# Git Commit Ajanı  (Ana Sınıf)
# ---------------------------------------------------------------------------

class GitCommitAgent:
    """
    Projedeki değişiklikleri analiz eden ve Conventional Commits standardına
    uygun atomik git commit'leri atan ajan.

    SRP: Yalnızca versiyon kontrolünden sorumludur.
    Diğer ajanlarla (ResearcherAgent, ValidatorAgent) doğrudan ilişkisi yoktur.

    Kullanım:
        agent = GitCommitAgent(repo_path="c:/Users/thinkpad/Desktop/caretta_track")
        summary = agent.auto_commit()
        print(summary)
    """

    def __init__(
        self,
        repo_path: str | Path | None = None,
        shell_tool: GitShellTool | None = None,
        classifier: ConventionalCommitClassifier | None = None,
        dry_run: bool = False,
        author_name: str = "",
        author_email: str = "",
    ) -> None:
        """
        Args:
            repo_path    : Git deposu kök dizini. None ise CWD kullanılır.
            shell_tool   : Enjekte edilebilir GitShellTool (DIP / test kolaylığı).
            classifier   : Enjekte edilebilir sınıflandırıcı (DIP).
            dry_run      : True ise git komutları çalıştırılmaz, yalnızca loglanır.
            author_name  : commit --author için ad. Boş ise git global config kullanılır.
            author_email : commit --author için e-posta.
        """
        resolved = Path(repo_path).resolve() if repo_path else Path.cwd()
        self._repo_path = resolved
        self._shell = shell_tool or GitShellTool(repo_path=resolved)
        self._classifier = classifier or ConventionalCommitClassifier()
        self._dry_run = dry_run
        self._author = (
            f"{author_name} <{author_email}>"
            if author_name and author_email
            else None
        )

        logger.info(
            "GitCommitAgent başlatıldı | Repo: %s | DryRun: %s",
            self._repo_path, self._dry_run,
        )

    # ------------------------------------------------------------------
    # Ana Giriş Noktası (izole, ana servise bağlı değil)
    # ------------------------------------------------------------------

    def auto_commit(
        self,
        paths: list[str] | None = None,
    ) -> AutoCommitSummary:
        """
        Değişiklikleri analiz eder ve atomik Conventional Commits atar.

        İşleyiş:
          1. Repo geçerliliğini kontrol et (fail-fast).
          2. git status → değişiklik listesi al.
          3. Değişiklik yoksa → NothingToCommitError (fail-fast).
          4. Sınıflandır → CommitGroup listesi.
          5. Her grup için: git add → git commit.
          6. AutoCommitSummary döndür.

        Args:
            paths: Yalnızca belirtilen yollardaki değişikliklere bak.
                   None ise repo'nun tamamı taranır.

        Returns:
            AutoCommitSummary: İşlem özeti.

        Raises:
            NothingToCommitError : Commitlenecek değişiklik yoksa.
            GitCommandError      : Git komutu başarısız olursa.
        """
        started_at = datetime.now()
        logger.info("═══ auto_commit başlıyor ═══ Repo: %s", self._repo_path)

        # 1. Repo kontrolü
        self._assert_valid_repo()

        # 2. Değişiklikleri al
        changes = self._get_changes(paths)

        # 3. Fail-fast: Değişiklik yoksa dur
        if not changes:
            logger.info("Commitlenecek değişiklik bulunamadı. İşlem sonlandırıldı.")
            raise NothingToCommitError(
                "Çalışma ağacında veya staging alanında değişiklik yok."
            )

        logger.info("%d değişiklik tespit edildi.", len(changes))

        # 4. Sınıflandır
        groups = self._classifier.classify(changes)

        # 5. Her grup için commit at
        results: list[CommitResult] = []
        for group in groups:
            result = self._stage_and_commit(group)
            results.append(result)

        finished_at = datetime.now()
        summary = AutoCommitSummary(
            started_at=started_at,
            finished_at=finished_at,
            repo_path=str(self._repo_path),
            total_changes=len(changes),
            groups_found=len(groups),
            commits_made=sum(1 for r in results if r.success),
            commits_failed=sum(1 for r in results if not r.success),
            results=results,
        )

        logger.info("auto_commit tamamlandı:\n%s", summary)
        return summary

    # ------------------------------------------------------------------
    # Yardımcı Metotlar (private)
    # ------------------------------------------------------------------

    def _assert_valid_repo(self) -> None:
        """
        Klasörün geçerli bir Git deposu olduğunu doğrular.
        Değilse → GitCommandError (fail-fast).
        """
        try:
            self._shell.run("rev-parse", "--git-dir")
            logger.debug("Git deposu doğrulandı: %s", self._repo_path)
        except GitCommandError as exc:
            logger.error("Geçerli bir git deposu değil: %s", self._repo_path)
            raise GitCommandError(
                "rev-parse --git-dir", exc.returncode,
                f"{self._repo_path} bir git deposu değil."
            ) from exc

    def _get_changes(self, paths: list[str] | None) -> list[ChangeRecord]:
        """
        `git status --porcelain` çıktısını parse eder.

        git status --porcelain örnek çıktısı:
            A  agents/researcher.py
            M  requirements.txt
            ?? untracked_file.py

        Returns:
            list[ChangeRecord]: İzlenen (tracked) tüm değişiklikler.
        """
        try:
            cmd_args = ["status", "--porcelain"]
            if paths:
                cmd_args.extend(["--"] + paths)

            raw = self._shell.run(*cmd_args)
        except GitCommandError as exc:
            logger.error("git status başarısız: %s", exc)
            raise

        if not raw:
            return []

        records: list[ChangeRecord] = []
        for line in raw.splitlines():
            if len(line) < 4:
                continue

            # Porcelain v1: "XY path" veya "XY old -> new" (rename)
            xy = line[:2]
            path_part = line[3:]

            # Rename: "R  old -> new" → yeni yolu al
            if " -> " in path_part:
                path_part = path_part.split(" -> ")[-1]

            # Staging area durumu (index): ilk karakter
            index_status = xy[0].strip()
            # Working tree durumu: ikinci karakter
            worktree_status = xy[1].strip()

            # Untracked (?) → git add ile staging'e alınabilir, yeni dosya olarak işle
            if xy.strip() == "??":
                records.append(ChangeRecord(status="A", path=path_part.strip()))
                continue

            # Staging'deki değişikliği tercih et; yoksa working tree'yi kullan
            effective_status = index_status or worktree_status or "M"
            records.append(ChangeRecord(status=effective_status, path=path_part.strip()))

        logger.debug("Çözümlenen değişiklikler: %s", records)
        return records

    def _stage_and_commit(self, group: CommitGroup) -> CommitResult:
        """
        Bir CommitGroup için dosyaları staging'e alır ve commit atar.

        Returns:
            CommitResult: Başarı/başarısızlık bilgisi.
        """
        message = group.message
        logger.info(
            "Commit hazırlanıyor: '%s' | Dosyalar: %s",
            message, group.files,
        )

        if self._dry_run:
            logger.info("[DRY RUN] Atlanıyor: %s", message)
            return CommitResult(
                success=True,
                message=message,
                files=group.files,
                sha="dry-run",
            )

        # --- git add ---
        try:
            self._shell.run("add", "--", *group.files)
            logger.debug("Staged: %s", group.files)
        except GitCommandError as exc:
            logger.error("git add başarısız (%s): %s", group.files, exc)
            return CommitResult(
                success=False,
                message=message,
                files=group.files,
                error=str(exc),
            )

        # --- git commit ---
        try:
            commit_args = ["commit", "-m", message]
            if self._author:
                commit_args += ["--author", self._author]

            self._shell.run(*commit_args)
            sha = self._get_latest_sha()
            logger.info("Commit atıldı: [%s] %s", sha, message)
            return CommitResult(success=True, message=message, files=group.files, sha=sha)

        except GitCommandError as exc:
            logger.error("git commit başarısız ('%s'): %s", message, exc)
            return CommitResult(
                success=False,
                message=message,
                files=group.files,
                error=str(exc),
            )

    def _get_latest_sha(self) -> str:
        """Son commit'in kısa SHA'sını döner."""
        try:
            return self._shell.run("log", "-1", "--format=%h")
        except GitCommandError:
            return "unknown"

    # ------------------------------------------------------------------
    # Yardımcı Araçlar (public utility)
    # ------------------------------------------------------------------

    def preview(self, paths: list[str] | None = None) -> list[CommitGroup]:
        """
        Commit atmadan hangi grupların oluşacağını gösterir.
        Hata ayıklama ve test için kullanılır.

        Returns:
            list[CommitGroup]: Oluşacak commit grupları.
        """
        self._assert_valid_repo()
        changes = self._get_changes(paths)
        if not changes:
            logger.info("Preview: Değişiklik bulunamadı.")
            return []
        return self._classifier.classify(changes)

    def get_status(self) -> dict[str, Any]:
        """
        Mevcut repo durumunu sözlük olarak döner.

        Returns:
            dict: branch, commit_count, pending_files, has_changes anahtarları.
        """
        try:
            branch = self._shell.run("rev-parse", "--abbrev-ref", "HEAD")
        except GitCommandError:
            branch = "unknown"

        changes = self._get_changes(None)
        return {
            "repo_path": str(self._repo_path),
            "branch": branch,
            "has_changes": len(changes) > 0,
            "pending_files": len(changes),
            "pending_details": [
                {"status": c.status, "path": c.path} for c in changes
            ],
        }
