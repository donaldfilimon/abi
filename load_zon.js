const loadZon = async (path) => {
  const res = await fetch(path, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to load ${path}`);
  const text = await res.text();
  let result = text.trim();
  if (result.startsWith(".{")) {
    result = result.replace(/^\.\{/, "[").replace(/\}$/, "]");
    result = result.replace(/\.\{/g, "{");
    result = result.replace(/\.[a-zA-Z0-9_]+ =/g, '"$1":');
    result = result.replace(/\.@\"(.*?)\" =/g, '"$1":');
    result = result.replace(/,(\s*[\}\]])/g, "$1");
  }
  return JSON.parse(result);
};
