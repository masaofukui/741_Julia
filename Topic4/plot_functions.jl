

function plt_fun(param,x,y; lw = 4, ymin=0,ymax=0, xlabel = "",title = "")
    @unpack_model param
    plt = plot(x,y,label=:none,lw=lw)
    plot!(xlabel=xlabel,title= title)
    plot!(titlefontfamily = "Computer Modern",
    xguidefontfamily = "Computer Modern",
    yguidefontfamily = "Computer Modern",
    legendfontfamily = "Computer Modern",
    titlefontsize=20,xguidefontsize=12,legendfontsize=12,yguidefontsize=12)
    if ymin != 0 && ymax != 0
        plot!(ylim=(ymin,ymax))
    end
    display(plt)
    return plt
end


function plt_by_age_fun(param,x,y; lw = 4, ymin=0,ymax=0, xlabel = "",title = "")
    @unpack_model param
    plt_by_age = plot()
    for ia in [25 20 15 10 5 1]
        age = Int(round(ag[ia]))
        plot!(x,y[:,ia],label="age = $ia",lw=3)
    end
    plot!()
    plot!(xlabel=xlabel,title= title)
    plot!(titlefontfamily = "Computer Modern",
    xguidefontfamily = "Computer Modern",
    yguidefontfamily = "Computer Modern",
    legendfontfamily = "Computer Modern",
    titlefontsize=20,xguidefontsize=12,legendfontsize=12,yguidefontsize=12)
    if ymin != 0 && ymax != 0
        plot!(ylim=(ymin,ymax))
    end
    display(plt_by_age)
    return plt_by_age
end
