#let course = "Stanford CS336"
#let title = "Assignment 1 Report"
#import "@preview/gentle-clues:1.2.0": *

#set text(
    font: ("Libertinus Serif", "Source Han Serif SC"),
    size: 11pt,
)

#set par(
    leading: 11pt,
    first-line-indent: 2.8em,
    justify: true,
)

#let fakepar = context [
    #let b = par[#box()]
    #let t = measure(b + b);
    #b
    #v(-t.height)
]

#set page(
    paper: "a4",
    header: [
        #set text(size: 10pt, baseline: 8pt, spacing: 3pt)
        #smallcaps[#course]
        #h(1fr) _ #title _
        #v(0.2em)
        #line(length: 100%, stroke: 0.7pt)
    ],
    footer: context [
        #set align(center)
        #grid(
            columns: (5fr, 1fr, 5fr),
            line(length: 100%, stroke: 0.7pt),
            text(size: 10pt, baseline: -3pt, counter(page).display("1")),
            line(length: 100%, stroke: 0.7pt),
        )
    ],
)

#show raw.where(block: false): box.with(
    fill: luma(240),
    inset: (x: 3pt, y: 0pt),
    outset: (y: 3pt),
    radius: 2pt,
)

#let style-number(number) = text(gray)[#number]
#show raw.where(block: true): it => {
    set text(font: "DejaVu Sans Mono", size: 8pt)
    set par(leading: 7pt)
    h(0em)
    v(-1.2em)
    block(
        width: 100%,
        fill: luma(240),
        inset: 10pt,
        radius: 10pt,
        grid(
            columns: (10pt, 400pt),
            align: (right, left),
            gutter: 0.5em,
            ..it.lines.enumerate().map(((i, line)) => (style-number(i + 1), line)).flatten()
        ),
    )
}

#set heading(numbering: (..args) => {
    let nums = args.pos()
    let level = nums.len()
    if level == 2 {
        numbering("1.", nums.at(1))
    } else if level == 3 {
        [#numbering("1.1", nums.at(1), nums.at(2))]
    }
    // } else if level == 4 {
    //     numbering("1.1.1", nums.at(1),nums.at(2),nums.at(3))
    // } else if level == 5 {
    //     [#numbering("1.1.1.1", nums.at(1),nums.at(2),nums.at(3),nums.at(4))]
    // }
})

#show heading.where(
    level: 1,
): it => {
    set align(center)
    set text(weight: "bold", size: 20pt, font: ("Libertinus Serif", "Source Han Serif SC"))
    it
    h(-0.4em)
}

#let line_under_heading() = {
    h(0em)
    v(-2.2em)
    line(length: 28%, stroke: 0.15pt)
}

#show heading.where(
    level: 2,
): it => {
    set text(weight: "bold", size: 17pt, font: ("Libertinus Serif", "Source Han Serif SC"))
    set block(above: 1.5em, below: 20pt)
    it
    line_under_heading()
    fakepar
}


#show heading.where(
    level: 3,
): it => {
    set text(weight: "bold", size: 13pt, font: ("Libertinus Serif", "Source Han Serif SC"))
    set block(above: 1.5em, below: 1em)
    it
}

#let thickness = 0.8pt
#let offset = 4pt
#let ubox(..) = box(
    width: 1fr,
    baseline: offset,
    stroke: (bottom: thickness),
)
#let uline(body) = {
    ubox()
    underline(
        stroke: thickness,
        offset: offset,
    )[#body]
    ubox()
}

#set list(
    marker: ([•], [▹], [–]),
    indent: 0.45em,
    body-indent: 0.5em,
)

#show list: it => {
    it
    fakepar
}

#show figure: it => {
    it
    h(0em)
    v(-1.2em)
}


// #show figure.caption: it => [
//     #set text(size: 8pt, font: "LXGW WenKai Mono")
//     图#it.counter.display(it.numbering)：#it.body
// ]

= CS336 Assignment 1 Writeup

== BPE Tokenization

=== Problem Unicode-1

#question(title: "Question 1")[What Unicode character does `chr(0)` return?]

#question(
    title: "Question 2",
)[How does this character's string representation (`__repr__()`) differ from its printed representation?]

#question(
    title: "Question 3",
)[What happens when this character occurs in text? It may be helpful to play around with the following in your Python interpreter and see if it matches your expectations:

    ```python
    >>> chr(0)
    >>> print(chr(0))
    >>> "this is a test" + chr(0) + "string"
    >>> print("this is a test" + chr(0) + "string")
    ```]

=== Problem Unicode-2

=== Problem `train_bpe`

=== Problem `train_bpe_tinystories`

Memory peak：931M

=== Problem `train_bpe_expts_owt`
