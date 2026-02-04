(function () {
    "use strict";
    document.documentElement.classList.add("js");

    // Theme toggle
    const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    const savedTheme = localStorage.getItem("theme");
    const initialTheme = savedTheme || (prefersDark ? "dark" : "light");

    document.documentElement.setAttribute("data-theme", initialTheme);

    function createThemeToggle() {
        const btn = document.createElement("button");
        btn.className = "theme-toggle";
        btn.setAttribute("aria-label", "Toggle theme");
        btn.textContent = initialTheme === "dark" ? "☀️ Light" : "🌙 Dark";

        btn.addEventListener("click", function () {
            const current = document.documentElement.getAttribute("data-theme");
            const next = current === "dark" ? "light" : "dark";
            document.documentElement.setAttribute("data-theme", next);
            localStorage.setItem("theme", next);
            btn.textContent = next === "dark" ? "☀️ Light" : "🌙 Dark";
        });

        document.body.appendChild(btn);
    }

    // Search functionality
    let searchIndex = null;

    function createSearch() {
        const sidebar = document.querySelector(".sidebar");
        if (!sidebar) return;

        const container = document.createElement("div");
        container.className = "search-container";

        const input = document.createElement("input");
        input.type = "text";
        input.className = "search-input";
        input.placeholder = "Search docs...";

        const results = document.createElement("div");
        results.className = "search-results";

        container.appendChild(input);
        container.appendChild(results);

        const tagline = sidebar.querySelector(".tagline");
        if (tagline) {
            tagline.insertAdjacentElement("afterend", container);
        } else {
            const brand = sidebar.querySelector(".brand");
            if (brand) {
                brand.insertAdjacentElement("afterend", container);
            }
        }

        input.addEventListener("input", function () {
            const query = this.value.trim().toLowerCase();
            if (query.length < 2) {
                results.classList.remove("active");
                return;
            }

            if (!searchIndex) {
                buildSearchIndex();
            }

            const matches = searchIndex.filter(function (item) {
                return item.title.toLowerCase().includes(query) ||
                       item.section.toLowerCase().includes(query);
            });

            // Clear results using safe DOM manipulation
            while (results.firstChild) {
                results.removeChild(results.firstChild);
            }

            if (matches.length === 0) {
                const noResult = document.createElement("div");
                noResult.className = "search-result-item";
                const noResultTitle = document.createElement("span");
                noResultTitle.className = "search-result-title";
                noResultTitle.textContent = "No results found";
                noResult.appendChild(noResultTitle);
                results.appendChild(noResult);
            } else {
                matches.forEach(function (item) {
                    const link = document.createElement("a");
                    link.href = item.url;
                    link.className = "search-result-item";

                    const title = document.createElement("div");
                    title.className = "search-result-title";
                    title.textContent = item.title;

                    const section = document.createElement("div");
                    section.className = "search-result-section";
                    section.textContent = item.section;

                    link.appendChild(title);
                    link.appendChild(section);
                    results.appendChild(link);
                });
            }

            results.classList.add("active");
        });

        document.addEventListener("click", function (e) {
            if (!container.contains(e.target)) {
                results.classList.remove("active");
            }
        });
    }

    function buildSearchIndex() {
        searchIndex = [];
        const navSections = document.querySelectorAll(".nav-section");

        navSections.forEach(function (section) {
            const sectionTitle = section.querySelector(".nav-title");
            const sectionName = sectionTitle ? sectionTitle.textContent : "";

            section.querySelectorAll(".nav-link").forEach(function (link) {
                searchIndex.push({
                    title: link.textContent,
                    section: sectionName,
                    url: link.getAttribute("href")
                });
            });
        });
    }

    // Initialize when DOM is ready
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", function () {
            createThemeToggle();
            createSearch();
        });
    } else {
        createThemeToggle();
        createSearch();
    }
})();
