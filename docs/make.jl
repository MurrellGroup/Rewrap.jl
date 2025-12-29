using Rewrap
using Documenter

DocMeta.setdocmeta!(Rewrap, :DocTestSetup, :(using Rewrap); recursive=true)

makedocs(;
    modules=[Rewrap],
    authors="Anton Oresten <antonoresten@proton.me> and contributors",
    sitename="Rewrap.jl",
    format=Documenter.HTML(;
        canonical="https://MurrellGroup.github.io/Rewrap.jl",
        edit_link="main",
        assets=String["assets/favicon.ico"],
    ),
    pages=[
        "Home" => "index.md",
        "API" => "api.md",
    ],
)

deploydocs(;
    repo="github.com/MurrellGroup/Rewrap.jl",
    devbranch="main",
)
