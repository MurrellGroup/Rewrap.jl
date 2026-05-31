module cuTileExt

using Rewrap
using cuTile

Rewrap.supports_fallback(::Type{<:cuTile.Tile}) = true

end
