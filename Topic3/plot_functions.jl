
function plt_fun(param,x,y; lw = 4, ymin=0,ymax=0, xlabel = "",title = "",x_baseline="none",y2="none",label1=:none,label2=:none)
    @unpack_model param
    plt = plot(x,y,label=label1,lw=lw)
    if y2 != "none"
        plot!(x,y2,label=label2,lw=lw,linestyle=:dash)
    end
    plot!(xlabel=xlabel,title= title)
    plot!(titlefontfamily = "Computer Modern",
    xguidefontfamily = "Computer Modern",
    yguidefontfamily = "Computer Modern",
    legendfontfamily = "Computer Modern",
    titlefontsize=20,xguidefontsize=12,legendfontsize=12,yguidefontsize=12)
    if ymin != 0 && ymax != 0
        plot!(ylim=(ymin,ymax))
    end

    if x_baseline != "none"
        vline!([x_baseline], label=:none,color=:red,linestyle=:dash)
        ymin, ymax = ylims(plt)
        xmin, xmax = xlims(plt)
        annotate!(x_baseline+0.01*(xmax-xmin), ymin+0.5*(ymax-ymin), text("Baseline", :left,"red","Computer Modern"))
    end
    display(plt)
    return plt
end


function plt_by_age_fun(param,x,y; lw = 4, ymin=0,ymax=0, xlabel = "",title = "",x_baseline="none")
    @unpack_model param
    
    plt_by_age = plot()
    for i_age in [25 20 15 10 5 1]
        index_age = closest_index(ag, i_age)
        plot!(x,y[:,index_age],label="age = $i_age",lw=lw)
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
    if x_baseline != "none"
        vline!([x_baseline], label=:none,color=:red,linestyle=:dash)
        ymin, ymax = ylims(plt)
        xmin, xmax = xlims(plt)
        annotate!(x_baseline+0.01*(xmax-xmin), ymin+0.5*(ymax-ymin), text("Baseline", :left,"red","Computer Modern"))
    end
    display(plt_by_age)
    return plt_by_age
end



