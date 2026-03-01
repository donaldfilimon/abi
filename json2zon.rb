require 'json'

def to_zon(obj, indent = 0)
  s = '  ' * indent
  case obj
  when Array
    if obj.empty?
      '.{}'
    else
      ".{\n" + obj.map { |v| s + '  ' + to_zon(v, indent + 1) }.join(",\n") + ",\n" + s + '}'
    end
  when Hash
    if obj.empty?
      '.{}'
    else
      ".{\n" + obj.map { |k, v| s + '  .@"' + k.to_s.gsub('\\', '\\\\').gsub('"', '\"') + '" = ' + to_zon(v, indent + 1) }.join(",\n") + ",\n" + s + '}'
    end
  when String
    '"' + obj.gsub('\\', '\\\\').gsub('"', '\"') + '"'
  when NilClass
    'null'
  when TrueClass
    'true'
  when FalseClass
    'false'
  else
    obj.to_s
  end
end

puts to_zon(JSON.parse(ARGF.read))