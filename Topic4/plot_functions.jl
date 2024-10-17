
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
    if maximum(y) - minimum(y) < 1e-4
        plot!(ylim=(minimum(y)-0.05,maximum(y)+0.05))
    end

    if x_baseline != "none"
        vline!([x_baseline], label=:none,color=:red,linestyle=:dash)
        ymin, ymax = ylims(plt)
        xmin, xmax = xlims(plt)
        annotate!(x_baseline+0.01*(xmax-xmin), ymin+0.8*(ymax-ymin), text("Baseline", :left,"red","Computer Modern"))
    end
    display(plt)
    return plt
end

function plot_IRF(param,y,ss0; tpre=10,tmax=0,y2=y,label1=:none,label2=:none,lw=4,title = "",ylabel="% deviation from initial s.s.")
    @unpack_model param
    if tmax != 0
        y = y[tg .<= tamx]
        y2 = y2[tg .<= tmax]
        tg = tg[tg .<= tmax]
    end
    if tpre != 0
        yplot = [ss0*ones(tpre+1);y]
        y2plot = [ss0*ones(tpre+1);y2]
        tgplot = [-tpre:1:0; tg]
    end

    plt = plot(tgplot,yplot,label=label1,lw=lw)
    if y2 != y
        plot!(tgplot,y2plot,label=label2,lw=lw,linestyle=:dash)
    end
    plot!(xlabel="Year",title= title,ylabel=ylabel)
    plot!(titlefontfamily = "Computer Modern",
    xguidefontfamily = "Computer Modern",
    yguidefontfamily = "Computer Modern",
    legendfontfamily = "Computer Modern",
    titlefontsize=20,xguidefontsize=12,legendfontsize=12,yguidefontsize=12)


    display(plt)
    return plt
end



